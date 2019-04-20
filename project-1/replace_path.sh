#!/bin/bash

find ./ -name "*.out.*" -exec sed -i '' -e 's#/data/#/Users/alexon/#g' {} \;