__version__ ="0.0.1"
print("Loading AI Lab...")

import os
from IPython.display import display,display_html
# From awesome dash intro repo by Kevin Mader
# A quick intro to Dash made for the PyData event in Zurich
# https://github.com/4QuantOSS/DashIntro 

# Can use Jupyter nbserverproxy extension (available at /.../proxy/<port>)

def show_app(app, port = 10001, 
             width = 700, 
             height = 350, 
             offline = False,
            in_binder = None):
    in_binder ='JUPYTERHUB_SERVICE_PREFIX' in os.environ if in_binder is None else in_binder
    if in_binder:
        base_prefix = '{}proxy/{}/'.format(os.environ['JUPYTERHUB_SERVICE_PREFIX'], port)
        url = 'https://hub.gke.mybinder.org{}'.format(base_prefix)
        app.config.requests_pathname_prefix = base_prefix
    else:
        url = 'http://localhost:%d' % port
    iframe = '<a href="{url}" target="_new">Open in new window</a><hr><iframe src="{url}" width={width} height={height}></iframe>'.format(url = url, 
                                                                                  width = width, 
                                                                                  height = height)

    iframe = '<a href="{url}" target="_new">Open in new window</a><hr>'.format(url = url, 
                                                                                  width = width, 
                                                                                  height = height)

    display_html(iframe, raw = True)
    if offline:
        app.css.config.serve_locally = True
        app.scripts.config.serve_locally = True
        
    return app.run_server(debug=False, # needs to be false in Jupyter
                          host = '0.0.0.0',
                          port=port)