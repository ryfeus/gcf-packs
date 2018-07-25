# gcf-packs
Library packs for google cloud functions

## How to start

Upload ZIP directly to your cloud function or to Google Cloud Storage

## Current packs

### Selenium Chromium

#### Intro

Selenium on Chromium. In fact - a ready-made tool for web scraping. For example, the demo now opens a random page in Wikipedia and sends its header.

Useful for web testing and scraping.

#### Demo

Current demo opens random page from wiki (https://en.wikipedia.org/wiki/Special:Random) and prints title. Keep in mind that you have to unpack https://github.com/ryfeus/gcf-packs/blob/master/selenium_chrome/source/headless-chromium.zip before modifying your package (github doesn't allow to upload files bigger than 100 mb)

#### Documentation

https://selenium-python.readthedocs.io/

#### Used code and documentation

https://github.com/adieuadieu/serverless-chrome
https://medium.com/clog/running-selenium-and-headless-chrome-on-aws-lambda-fb350458e4df