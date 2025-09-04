from urllib.parse import unquote
from flask import Flask, request, make_response
from flask import send_file
from io import BytesIO
from PIL import Image
import json
import requests
import math
import re
import gzip, netCDF4
import asyncio
from aiohttp import ClientSession
import base64
import matplotlib
from dateutil.rrule import rrulestr
from datetime import timezone
from datetime import datetime as dt
from dateutil.tz import gettz
from utils.utils import reload_module
reload_module('heatmap_grid')
reload_module('heatmap_footprint')
from heatmap_footprint import Footprint
from timezonefinder import TimezoneFinder
import logging
import yaml
import tempfile
import rasterio
from rasterio.transform import from_bounds
from pyproj import Transformer
import numpy as np

# TODO: User can set output image size?
MAP_WIDTH_PIXELS = 640
MAP_HEIGHT_PIXELS = 597

NUM_REQUESTS_IN_PARALLEL = 25
MAX_NUM_DATES_TO_PARSE = 750

CITIES_DATA_URL = "https://storage.googleapis.com/air-tracker-edf-website/prod/assets/data/cities/cities.json"

app = Flask(__name__)

# TODO: Store in a different way?
credentials = json.load(open("credentials.json"))

# Logging setup
logging.basicConfig(filename='logging.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


# Any error we don't explicitly catch in our code falls to here
@app.errorhandler(Exception)
def exception_fallback_handler(error):
    error_message = str(error.args[0]) if error.args else str(error)
    logging.exception("Unhandled exception caught in Flask error handler")
    if 'day is out of range' in error_message or 'does not match format' in error_message:
        return return_error("Error parsing dates. Check that list contains valid dates.")
    elif 'operands could not be broadcast together with shapes' in error_message:
        return return_error("Discrepancy between footprint sizes for chosen dates. Please send the list of dates to an EDF contact for analysis. In the meantime, try again with a different set of dates.")
    else:
        return return_error("Unknown error occurred. Please reach out to support.")


@app.route('/docs')
def redirect_to_docs():
    # load iframe of https://storage.googleapis.com/air-tracker-edf-website/prod/assets/data/api.html
    return requests.get("https://storage.googleapis.com/air-tracker-edf-website/prod/assets/data/api.html").content

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/favicon.ico')
def ignore_favicon():
    return '', 204


@app.route('/')
def api_description():
    return 'Available API calls: \n /get_footprint \n /get_heatmap'


@app.route('/get_footprint')
def get_footprint():
    view = request.args.get('view') or request.args.get('v')
    if not view:
        return return_error("view parameter (view=lat,lon,zoom) is required.")

    view_obj = view.split(",")
    if len(view_obj) < 3:
        return return_error("Malformed view parameter. Required format: (view=lat,lon,zoom)")

    [map_center, map_zoom, latOffset, lonOffset] = get_formatted_view(view_obj)

    # Dates come in is ISO8601 format
    time = request.args.get('time') or request.args.get('t')
    tz = request.args.get('tz')
    format = request.args.get("format", "png").lower()
    is_real_time = request.args.get('isRealTime', 'false').lower() == 'true'

    if not tz:
        tz_finder = TimezoneFinder()
        tz = tz_finder.timezone_at(lat=map_center['lat'], lng=map_center['lon'])

    if not time:
        return return_error("time parameter (time=YYYY-MM-DDTHH:MM) is required.")
    elif re.match(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}$", time) is None:
        return return_error("Malformed time parameter. Required format: (time=YYYY-MM-DDTHH:MM)")
    elif tz is None:
        return return_error(f"When passing in a list of dates, tz info is required (e.g. America/New_York)")

    # Convert to UTC (based on the tz info passed in) and also format to the expected format of footprints stored on GCS.
    time = dt.strftime(dt.strptime(time, '%Y-%m-%dT%H:%M').replace(tzinfo=gettz(tz)).astimezone(timezone.utc), "%Y-%m-%dT%H-00-00")

    show_basemap = request.args.get('showBasemap', default=True, type=lambda v: v.lower() == 'true')
    show_contribution_likelihoods = request.args.get('showContributionLikelihoods', default=False, type=lambda v: v.lower() == 'true')

    cities_data = requests.get(CITIES_DATA_URL).json()["cities"]
    domain = None
    for city in cities_data.values():  # use .values() to get the dicts
        bounds = city.get("click_bounds")
        if bounds and is_in_bounds(map_center['lat'], map_center['lon'], bounds):
            domain = city["domain_name"]
            break

    if is_real_time and not domain:
        return return_error("Domain data not available for the provided lat,lon.")

    if format == "netcdf":
        if is_real_time:
            formatted_time = parse_time_for_realtime(time)
            footprint_url = f"https://api2.airtracker.createlab.org/get_footprint?lat={map_center['lat']}&lon={map_center['lon']}&time={formatted_time}&region={domain}&asNetCDF=true"
            logging.info(f"Fetching footprint from: {footprint_url}")

            footprint_netcdf = read_base64_netcdf_footprint(requests.get(footprint_url).content, return_bytes=True)
        else:
            footprint_url = f"https://storage.googleapis.com/download/storage/v1/b/air-tracker-edf-prod/o/by-simulation-id%2F{time}%2F{map_center['lon']}%2F{map_center['lat']}%2F1%2Ffootprint.nc.gz?alt=media"

            response = requests.get(footprint_url, stream=True)
            response.raise_for_status()
            footprint_netcdf = response.raw

        return send_file(
            footprint_netcdf,
            mimetype="application/x-netcdf",
            as_attachment=True,
            download_name="footprint.nc"
        )

    map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={map_center['lat']},{map_center['lon']}&zoom={map_zoom}&size={MAP_WIDTH_PIXELS}x{MAP_HEIGHT_PIXELS}&scale=1&key={credentials['google_api']}"

    if is_real_time:
        formatted_time = parse_time_for_realtime(time)
        footprint_url = f"https://api2.airtracker.createlab.org/get_footprint?lat={map_center['lat']}&lon={map_center['lon']}&time={formatted_time}&region={domain}"
        logging.info(f"Fetching footprint from: {footprint_url}")
        try:
            response = requests.get(footprint_url)
            response.raise_for_status()
            decoded_json = json.loads(response.content.decode('utf-8'))
            header, encoded_footprint = decoded_json["data"]["mediaLink"].split(",", 1)
            image_data = base64.b64decode(encoded_footprint)
            original_footprint = Image.open(BytesIO(image_data))
            footprint_metadata = {"metadata": decoded_json["data"]["metadata"]}
        except requests.exceptions.HTTPError as e:
            logging.error(e)
            return return_error("Invalid footprint lat,lon provided or footprint not available at that location or specific time.")
    else:
        footprint_url = f"https://storage.googleapis.com/air-tracker-edf-prod/by-simulation-id/{time}/{map_center['lon']}/{map_center['lat']}/1/footprint.png"
        footprint_metadata_url = f"https://storage.googleapis.com/storage/v1/b/air-tracker-edf-prod/o/by-simulation-id%2F{time}%2F{map_center['lon']}%2F{map_center['lat']}%2F1%2Ffootprint.png"
        try:
            response = requests.get(footprint_url, stream=True)
            response.raise_for_status()
            footprint_img = response.raw
            original_footprint = Image.open(footprint_img)
            footprint_metadata = requests.get(footprint_metadata_url, stream=True).json()
        except requests.exceptions.HTTPError as e:
            logging.error(e)
            return return_error("Invalid footprint lat,lon provided or footprint not available at that location or specific time.")

    original_pixel_map = original_footprint.load()

    # TODO: Check if Google response valid?
    base_map = requests.get(map_url, stream=True).raw if show_basemap else None

    if show_contribution_likelihoods: # Use original footprint from Cloud Storage
        new_footprint = original_footprint
    else:  # Create masked footprint
        new_footprint = Image.new(original_footprint.mode, original_footprint.size)

        new_pixel_map = new_footprint.load()
        # Set 80% transparency
        finalTransparency = 0.8

        for x in range(new_footprint.size[0]):
            for y in range(new_footprint.size[1]):
                r = original_pixel_map[x,y][0]
                g = original_pixel_map[x,y][1]
                b = original_pixel_map[x,y][2]
                a = original_pixel_map[x,y][3]
                if not show_contribution_likelihoods:
                    # Color saturation
                    s = max(r, g, b) - min(r, g, b)
                    r = g = b = 200
                    if (a < 128):
                        # Outside backtrace; grey overlay
                        a = int(255 * finalTransparency)
                    else:
                        # Inside backtrace; alpha goes to zero as saturation increases
                        gain = 14 # higher gain means sharper transition
                        a = max(0, 255 - s * gain)
                new_pixel_map[x,y] = (r,g,b,a)

    if format == "geotiff":
        # Use the RGBA footprint directly (no warping)
        image_rgba = np.array(new_footprint.convert("RGBA"))
        height, width, _ = image_rgba.shape

        # Original footprint bounds + offsets
        xmin = float(footprint_metadata["metadata"]["xmin"]) + lonOffset
        xmax = float(footprint_metadata["metadata"]["xmax"]) + lonOffset
        ymin = float(footprint_metadata["metadata"]["ymin"]) + latOffset
        ymax = float(footprint_metadata["metadata"]["ymax"]) + latOffset

        # Transform bounds to Web Mercator
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xmin_3857, ymin_3857 = transformer.transform(xmin, ymin)
        xmax_3857, ymax_3857 = transformer.transform(xmax, ymax)

        transform = from_bounds(xmin_3857, ymin_3857, xmax_3857, ymax_3857, width, height)

        memfile = rasterio.MemoryFile()
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=4,
            dtype='uint8',
            transform=transform,
            crs="EPSG:3857",
            photometric='RGB'
        ) as dataset:
            for i in range(4):
                dataset.write(image_rgba[:, :, i], i + 1)

        return send_file(
            memfile,
            mimetype="image/tiff",
            as_attachment=True,
            download_name="footprint.tif"
        )

    transformed_overlay = get_transformed_overlay(map_center, map_zoom, footprint_metadata, new_footprint, show_contribution_likelihoods, latOffset, lonOffset)

    final_output = transformed_overlay

    if show_basemap:
        background = Image.open(base_map).convert('RGBA')
        if show_contribution_likelihoods:
            # Add opacity to overlay; modifies passed in image
            alter_overlay_opacity(transformed_overlay, 0.80)
        # composite new overlay with the background
        final_output = Image.alpha_composite(background, transformed_overlay)

    return serve_image_from_memory(image=final_output)


@app.route('/get_heatmap')
async def get_heatmap():
    view = request.args.get('view') or request.args.get('v')
    if not view:
        return return_error("view parameter (view=lat,lon,zoom) is required.")

    view_obj = view.split(",")
    if len(view_obj) < 3 or len(view_obj) > 3:
        return return_error("Malformed view parameter. Required format: (view=lat,lon,zoom)")

    [map_center, map_zoom, latOffset, lonOffset] = get_formatted_view(view_obj)

    is_real_time = request.args.get('isRealTime', 'false').lower() == 'true'
    format = request.args.get("format", "png").lower()

    # Dates come in is ISO8601 format
    times = request.args.get('times') or request.args.get('ts')
    # We also support rrule expression to generate list of dates on the server side
    rrule = request.args.get('rrule')
    timestamps = []

    cities_data = requests.get(CITIES_DATA_URL).json()["cities"]
    domain = None
    for city in cities_data.values():  # use .values() to get the dicts
        bounds = city.get("click_bounds")
        if bounds and is_in_bounds(map_center['lat'], map_center['lon'], bounds):
            domain = city["domain_name"]
            break

    if is_real_time and not domain:
        return return_error("Domain data not available for the provided lat,lon.")

    if times:
        tz = request.args.get('tz')

        if not tz:
            tz_finder = TimezoneFinder()
            tz = tz_finder.timezone_at(lat=map_center['lat'], lng=map_center['lon'])

        regex_date_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}"
        if re.match(fr"^({regex_date_pattern})(,{regex_date_pattern})+,?$", times) is None:
            return return_error("Malformed times parameter. Required format: List of DateTimes (times=YYYY-MM-DDTHH:MM,...) with minimum of two DateTimes.")
        elif tz is None:
            return return_error("When passing in a list of dates, tz info is required (e.g. America/New_York)")

        # Create a datetime string list from the query string, removing duplicates from it (via the set method)
        tmp_times_list = list(set(times.replace(" ", "").split(",")))

        if len(tmp_times_list) > MAX_NUM_DATES_TO_PARSE:
            return return_error(f"Too many dates passed in. Max of {MAX_NUM_DATES_TO_PARSE} is allowed.")

        # Convert to UTC (based on the tz info passed in) and also format to the expected format of footprints stored on GCS.
        for date in tmp_times_list:
            # It's possible we have an empty date after the above split because of a trailing comma at the end of the passed in date string
            if date:
                timestamps.append(dt.strftime(dt.strptime(date, '%Y-%m-%dT%H:%M').replace(tzinfo=gettz(tz)).astimezone(timezone.utc), "%Y-%m-%dT%H-00-00"))
    elif rrule:
        # TODO: Check validity of RRule
        rrule = unquote(rrule)
        # DSTART is timezone aware, so we convert to UTC for lookups on GSC
        timestamps = list(dt.strftime(date.astimezone(timezone.utc), "%Y-%m-%dT%H-00-00") for date in rrulestr(rrule))

        if len(timestamps) > MAX_NUM_DATES_TO_PARSE:
            return return_error(f"Too many dates passed in. Max of {MAX_NUM_DATES_TO_PARSE} is allowed.")
    else:
        return return_error("times parameter (times=YYYY-MM-DDTHH:MM,...) or RRULE is required.")

    show_basemap = request.args.get('showBasemap', default=True, type=lambda v: v.lower() == 'true')

    map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={map_center['lat']},{map_center['lon']}&zoom={map_zoom}&size={MAP_WIDTH_PIXELS}x{MAP_HEIGHT_PIXELS}&scale=1&key={credentials['google_api']}"

    for_overlay = request.args.get('forOverlay')

    # Get all files
    async with ClientSession() as session:
      netcdf_files = await gather_with_concurrency(NUM_REQUESTS_IN_PARALLEL, *[get_and_read_netcdf_footprint(time=timestamp, location=map_center, session=session, domain=domain, is_real_time=is_real_time) for timestamp in timestamps])

    num_processed_timestamps = 0
    first = True
    mean_dataset = None
    sum = None

    for idx, netcdf in enumerate(netcdf_files):
        if netcdf is None:
            continue

        footprint = netcdf.variables['foot'][:]

        if first:
            # TODO: xmin|xmax|ymin|ymax in the metadata is only written out to the footprint and not stored in the netcdf. It's pulled from simulation.config section of domains.yaml
            # Once the live EDF site is updated, we can use the domain_config logic that realtime uses
            if not is_real_time:
                footprint_metadata_url = f"https://storage.googleapis.com/storage/v1/b/air-tracker-edf-prod/o/by-simulation-id%2F{timestamps[idx]}%2F{map_center['lon']}%2F{map_center['lat']}%2F1%2Ffootprint.png"
            sum = footprint
            mean_dataset = netcdf
            first = False
        else:
            sum += footprint
        num_processed_timestamps += 1

    if num_processed_timestamps == 0:
        return return_error("Invalid lat,lon provided or no footprints available at that location or time steps.")

    mean = sum / num_processed_timestamps
    mean_dataset.variables['foot'] = mean

    if format == "netcdf":
        with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
            new_ds = netCDF4.Dataset(tmp.name, mode='w', format='NETCDF4')

            # --- Copy global attributes ---
            if hasattr(mean_dataset, 'ncattrs'):
                for attr_name in mean_dataset.ncattrs():
                    new_ds.setncattr(attr_name, mean_dataset.getncattr(attr_name))

            # --- Copy dimensions ---
            if hasattr(mean_dataset, 'dimensions'):
                for name, dimension in mean_dataset.dimensions.items():
                    new_ds.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

            # --- Copy variables ---
            for name, variable in mean_dataset.variables.items():
                if hasattr(variable, 'dimensions'):
                    # netCDF4.Variable path
                    dtype = variable.dtype
                    dims = variable.dimensions
                    data = mean if name == 'foot' else variable[:]
                else:
                    # MaskedArray or ndarray path
                    data = mean if name == 'foot' else variable
                    dtype = data.dtype
                    dims = tuple(f"dim_{i}" for i in range(data.ndim))

                    # Ensure fabricated dims exist
                    for i, size in enumerate(data.shape):
                        if dims[i] not in new_ds.dimensions:
                            new_ds.createDimension(dims[i], size)

                # Ensure dimensions exist before creating variable
                for dim in dims:
                    if dim not in new_ds.dimensions:
                        if hasattr(mean_dataset, 'dimensions') and dim in mean_dataset.dimensions:
                            new_ds.createDimension(dim, len(mean_dataset.dimensions[dim]))
                        else:
                            # Fallback size from data
                            idx = dims.index(dim)
                            new_ds.createDimension(dim, data.shape[idx])

                new_var = new_ds.createVariable(name, dtype, dims, zlib=True)

                if hasattr(variable, 'ncattrs'):
                    for attr_name in variable.ncattrs():
                        new_var.setncattr(attr_name, variable.getncattr(attr_name))

                new_var[:] = data

            new_ds.close()
            tmp.seek(0)
            buf = BytesIO(tmp.read())

        return send_file(
            buf,
            mimetype="application/x-netcdf",
            as_attachment=True,
            download_name="heatmap.nc"
        )


    #colors = ['#F9F9F9', '#FFDB9D', '#FE9367', '#D7456D', '#7E2482', '#430F75']
    #colors = ['#F9F9F9', '#07EFF7', '#658CF8', '#7C14F5', '#B71AF5', '#FB23F5']
    colors = ['#F9F9F9','#07EFF7','#658CF8','#658CF8','#7C14F5','#7C14F5','#B71AF5','#B71AF5','#FB23F5','#FB23F5','#FB23F5']
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', colors)
    #cmap = "BuPu"

    if request.args.get('colormapList'):
      tmp = request.args.get('colormapList').replace(" ", "").split(",")
      if (len(tmp) > 1):
        colors =  ["#" + color for color in tmp]
        #cmap = matplotlib.colors.ListedColormap(colors, name='custom_cmap')
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', colors)

    image_bytes = Footprint(mean_dataset).create_image(log10=True, cmap=cmap, vmin=-4, vmax=0, format=format)

    if is_real_time:
        domain_config = load_configmap_from_url("https://storage.googleapis.com/air-tracker-edf-website/prod/assets/data/cities/domains.yaml")
        footprint_metadata = domain_config["data"][f"{domain}.yaml"]["simulation_config"]
        # Website expect xmin not xmn, etc
        key_map = {
            "xmn": "xmin",
            "xmx": "xmax",
            "ymn": "ymin",
            "ymx": "ymax"
        }
        for old_key, new_key in key_map.items():
            if old_key in footprint_metadata:
                footprint_metadata[new_key] = footprint_metadata.pop(old_key)
        footprint_metadata = {'metadata': footprint_metadata}
    else:
        footprint_metadata = requests.get(footprint_metadata_url, stream=True).json()

    if for_overlay:
        b64_string = f'data:image/png;base64,' + base64.b64encode(image_bytes).decode('utf8')
        content = {'metadata': footprint_metadata['metadata'], 'image': b64_string}
        gzipped_content = gzip.compress(json.dumps(content).encode('utf8'), 5)
        response = make_response(gzipped_content)
        response.headers['Content-length'] = len(gzipped_content)
        response.headers['Content-Encoding'] = 'gzip'
        return response
    elif format == "geotiff":
        return serve_image_from_memory(
            image=image_bytes,
            is_bytes=True,
            mimetype='image/tiff',
            download_name='heatmap.tif'
        )
    else:
        new_footprint = Image.open(BytesIO(image_bytes))

        # TODO: Check if Google response valid?
        base_map = requests.get(map_url, stream=True).raw if show_basemap else None

        transformed_overlay = get_transformed_overlay(map_center, map_zoom, footprint_metadata, new_footprint, True, latOffset, lonOffset)

        final_output = transformed_overlay

        if show_basemap:
            background = Image.open(base_map).convert('RGBA')
            # Add opacity to overlay; modifies passed in image
            alter_overlay_opacity(transformed_overlay, 0.80)
            # composite new overlay with the background
            final_output = Image.alpha_composite(background, transformed_overlay)

        return serve_image_from_memory(image=final_output)


# coroutines
async def gather_with_concurrency(n, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro
    return await asyncio.gather(*(sem_coro(c) for c in coros))


def parse_time_for_realtime(time):
    date_part, time_part = time.split('T')
    year, month, day = date_part.split('-')
    hour = time_part.split('-')[0]
    formatted_time = year + month + day + hour
    return formatted_time


def get_formatted_view(view_obj):
    input_lat = float(view_obj[0])
    input_lon = float(view_obj[1])
    input_zoom = int(view_obj[2])

    latTrunc = str(round(input_lat,2))
    # Footprints are being stored in the bucket by lat/lon with 2 digit precision. Sorta.
    # If the two digit precision ends with trailing zero in the second digit, it is removed.
    # Note: No need to deal with this, since in python round will not zero pad the end, unlike in JS with toFixed
    # if (latTrunc.split(".")[1][1] == "0"):
    #     latTrunc = latTrunc[:-1]
    latOffset = input_lat - float(latTrunc)

    lonTrunc = str(round(input_lon, 2))
    # Footprints are being stored in the bucket by lat/lon with 2 digit precision. Sorta.
    # If the two digit precision ends with trailing zero in the *second* digit, it is removed.
    # Note: No need to deal with this, since in python round will not zero pad the end, unlike in JS with toFixed
    # if (lonTrunc.split(".")[1][1] == "0"):
    #     lonTrunc = lonTrunc[:-1]
    lonOffset = input_lon - float(lonTrunc)

    map_center = {"lat": float(latTrunc), "lon": float(lonTrunc)}
    map_zoom = input_zoom
    return [map_center, map_zoom, latOffset, lonOffset]


def alter_overlay_opacity(overlay, opacity):
    # Create a copy of the overlay
    tmp = overlay.copy()
    # Put alpha on the copy at 80%
    tmp.putalpha(round(255 * opacity))
    # merge overlay with mask
    overlay.paste(tmp, overlay)


def get_transformed_overlay(map_center, map_zoom, footprint_metadata, new_footprint, show_contribution_likelihoods, latOffset, lonOffset):
    center_mercator = lonlat_to_pixel_xy(map_center['lon'], map_center['lat'])
    map_mercator = {
        "west":  center_mercator[0] - (MAP_WIDTH_PIXELS / 2.0) / (2 ** map_zoom),
        "east":  center_mercator[0] + (MAP_WIDTH_PIXELS / 2.0) / (2 ** map_zoom),
        "north": center_mercator[1] - (MAP_HEIGHT_PIXELS / 2.0) / (2 ** map_zoom),
        "south": center_mercator[1] + (MAP_HEIGHT_PIXELS / 2.0) / (2 ** map_zoom)
    }

    # Convert to pixel coords. Also, handle offsets after truncating from original location in get_formatted_view()
    overlay_mercator = {
        "west": lonlat_to_pixel_xy(float(footprint_metadata["metadata"]["xmin"]) + lonOffset, float(footprint_metadata["metadata"]["ymin"]) + latOffset)[0],
        "east": lonlat_to_pixel_xy(float(footprint_metadata["metadata"]["xmax"]) + lonOffset, float(footprint_metadata["metadata"]["ymax"]) + latOffset)[0],
        "north": lonlat_to_pixel_xy(float(footprint_metadata["metadata"]["xmax"]) + lonOffset, float(footprint_metadata["metadata"]["ymax"]) + latOffset)[1],
        "south": lonlat_to_pixel_xy(float(footprint_metadata["metadata"]["xmin"]) + lonOffset, float(footprint_metadata["metadata"]["ymin"]) + latOffset)[1]
    }

    overlay_width_pixels = new_footprint.width
    overlay_height_pixels = new_footprint.height

    # Transform map mercator bounds into overlay pixel locations
    west_px = one_dim_lin_xform(map_mercator["west"], overlay_mercator["west"], overlay_mercator["east"], 0, overlay_width_pixels)
    east_px = one_dim_lin_xform(map_mercator["east"], overlay_mercator["west"], overlay_mercator["east"], 0, overlay_width_pixels)
    north_px = one_dim_lin_xform(map_mercator["north"], overlay_mercator["north"], overlay_mercator["south"], 0, overlay_height_pixels)
    south_px = one_dim_lin_xform(map_mercator["south"], overlay_mercator["north"], overlay_mercator["south"], 0, overlay_height_pixels)

    transformed_overlay = new_footprint.transform(
        (MAP_WIDTH_PIXELS, MAP_HEIGHT_PIXELS), # output size
        Image.QUAD,
        (
            west_px, north_px, # upper left
            west_px, south_px, # lower left
            east_px, south_px, # lower-right
            east_px, north_px  # upper-right
        ),
        resample=Image.BILINEAR,
        fill = 1,
        # Fillcolor only used when overlay doesn't completely cover the map area
        # TODO: why does c8c8c8cc (r=g=b=200, a=0.8*255) not actually match? Manually calculated a color that appears to essentially match.
        fillcolor = None if show_contribution_likelihoods else (160, 160, 160, 204)
    )
    return transformed_overlay


def return_error(msg):
    response = app.response_class(
        response=json.dumps({"error" : msg}),
        status=400,
        mimetype='application/json'
    )
    return response


def serve_image_from_memory(image=None, is_bytes=False, format='PNG', mimetype='image/png', download_name=None):
    if is_bytes:
        img_io = BytesIO(image)
    else:
        img_io = BytesIO()
        image.save(img_io, format)
    img_io.seek(0)
    return send_file(
        img_io,
        mimetype=mimetype,
        as_attachment=bool(download_name),
        download_name=download_name
    )


# Transform x from the space defined by from_min-from_max into the space defined by to_min-to_max
def one_dim_lin_xform(x: float, from_min: float, from_max: float, to_min: float, to_max: float):
    return to_min + (x - from_min) * (to_max - to_min) / (from_max - from_min)


# Convert between lat lon (degrees) and web mercator 0-256
def lonlat_to_pixel_xy(lon, lat):
    x = (lon + 180.0) * 256.0 / 360.0
    y = (256.0/2.0) - math.log(math.tan((
        lat + 90.0) * math.pi / 360.0)) * (
                256.0/2.0) / math.pi
    return [x, y]


# Web Mercator to Lon Lat
def pixel_xy_to_lonlat(x, y):
    lat = math.atan(math.exp(((256.0/2.0) - y) * math.pi / (
        256.0/2.0))) * 360.0 / math.pi - 90.0
    lon = x * 360.0 / 256.0- 180.0
    return {"lon":lon, "lat":lat}


async def get_and_read_netcdf_footprint(netcdf_footprint_url=None, time=None, location=None, session=None, domain=None, is_real_time=False):
    netcdf_data = await get_netcdf_footprint(netcdf_footprint_url, time, location, session, domain, is_real_time)
    if netcdf_data is None:
        return None

    if is_real_time:
        return read_base64_netcdf_footprint(netcdf_data)
    else:
        return read_compressed_netcdf_footprint(netcdf_data)


async def get_netcdf_footprint(netcdf_footprint_url=None, time=None, location=None, session=None, domain=None, is_real_time=False):
  if not netcdf_footprint_url:
    if is_real_time:
        formatted_time = parse_time_for_realtime(time)
        netcdf_footprint_url = f"https://api2.airtracker.createlab.org/get_footprint?lat={location['lat']}&lon={location['lon']}&time={formatted_time}&region={domain}&asNetCDF=true"
    else:
        netcdf_footprint_url = f"https://storage.googleapis.com/download/storage/v1/b/air-tracker-edf-prod/o/by-simulation-id%2F{time}%2F{location['lon']}%2F{location['lat']}%2F1%2Ffootprint.nc.gz?alt=media"
  else:
    assert(not session)
    assert(not time)
    assert(not location)

  logging.info(f"Fetching netcdf footprint from: {netcdf_footprint_url}")

  async with session.get(netcdf_footprint_url) as response:
    try:
      response.raise_for_status()
      return await response.read()
    except Exception as e:
      return None


def read_base64_netcdf_footprint(base64_netcdf_footprint, return_bytes=False):
    decoded_bytes = base64.b64decode(json.loads(base64_netcdf_footprint.decode('utf-8'))["data"]["mediaLink"])
    if return_bytes:
        return BytesIO(decoded_bytes)
    return netCDF4.Dataset('footprint.nc', memory=decoded_bytes)


def read_compressed_netcdf_footprint(compressed_netcdf_footprint):
    uncompressed = gzip.decompress(compressed_netcdf_footprint)
    return netCDF4.Dataset('footprint.nc', memory=uncompressed)


def load_configmap_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request failed

    # Load the top-level YAML
    configmap = yaml.safe_load(response.text)

    # Parse nested YAML strings in the 'data' section
    full_config = dict(configmap)
    data_section = configmap.get("data", {})
    parsed_data = {}

    for key, yaml_string in data_section.items():
        try:
            parsed_data[key] = yaml.safe_load(yaml_string)
        except yaml.YAMLError as e:
            parsed_data[key] = None

    full_config["data"] = parsed_data
    return full_config

def is_in_bounds(lat, lon, bounds):
    """Check if a point is within the given bounds dictionary."""
    return bounds["ymin"] <= lat <= bounds["ymax"] and bounds["xmin"] <= lon <= bounds["xmax"]


# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=8080, debug=True)
