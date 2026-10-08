#!/bin/sh
# Start of the bundled Nginx: HTTPS when the certificate files are there, plain HTTP otherwise.
if [ -f /etc/nginx/certs/bct.crt ] && [ -f /etc/nginx/certs/bct.key ]; then
    cp /etc/nginx/bct/https.conf /etc/nginx/conf.d/default.conf
    echo "BCT: HTTPS with the certificate deploy/nginx/certs/bct.crt"
else
    cp /etc/nginx/bct/http.conf /etc/nginx/conf.d/default.conf
    echo "BCT WARNING: no certificate (deploy/nginx/certs/bct.crt and bct.key): plain HTTP, passwords are not encrypted"
fi
exec nginx -g 'daemon off;'
