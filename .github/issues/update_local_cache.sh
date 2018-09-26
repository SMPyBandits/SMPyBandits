#!/usr/bin/env bash
wget -np -k -e robots=off -r -l 1 https://github.com/SMPyBandits/SMPyBandits/{milestones,issues,labels}/
