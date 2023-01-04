import multiprocessing

workers = 3
bind = 'unix:airtrackerapi.sock'
#umask = 0o002
reload = True
#worker_class = 'uvicorn.workers.UvicornWorker'


#logging
accesslog = '-'
errorlog = 'airtrackerapi-error.log'
