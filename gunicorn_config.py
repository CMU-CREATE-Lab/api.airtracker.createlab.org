import multiprocessing

workers = 6
# 2 minutes
timeout = 120
bind = 'unix:airtrackerapi.sock'
reload = True
# 4094 is default and the range is a number from 0 (unlimited) to 8190.
# Unless we do POSTs back to this service, our GET requests will fail with too
# long a URI due to the number of potential timestamps being sent for processing.
# Note that we also have to modify our webserver config (Apache in our hal50 case)
# to expand the allowed value of both 'LimitRequestLine' and 'LimitRequestFieldSize'
# to a value of 16376 (4x default value)
limit_request_line = 0
# Logging
accesslog = '-'
errorlog = 'airtrackerapi-error.log'
