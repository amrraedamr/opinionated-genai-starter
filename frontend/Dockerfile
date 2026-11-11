# The test console: static files plus a proxy to each service on one origin.
FROM nginx:1.29-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY proxy.inc /etc/nginx/snippets/proxy.inc
COPY index.html console.css console.js mic-worklet.js /usr/share/nginx/html/

EXPOSE 8080
