#!/usr/bin/env bash
if [ $LOAD_GLOBAL_SOURCES = "1" ]; then
    cp -r $1/../../verilog/cfu.v .
    cp -r $1/../../verilog/verilog_src/ .
    cp -r $1/../../acceleration_src/src/ .
fi
