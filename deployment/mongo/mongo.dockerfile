FROM mongo:7.0.2

COPY init.js /docker-entrypoint-initdb.d/
