FROM node:16.13.0-alpine AS build

WORKDIR /build

COPY package.json package.json
COPY package-lock.json package-lock.json
RUN npm cache clean --force
RUN npm config set fetch-retry-mintimeout 20000
RUN npm config set fetch-retry-maxtimeout 120000
RUN npm ci
RUN npm install vis-data

COPY tsconfig.json tsconfig.json
COPY public public
COPY src src
RUN npm run build
ENV CI=true
RUN npm test

FROM nginx:1.21.3-alpine

COPY --from=build /build/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
