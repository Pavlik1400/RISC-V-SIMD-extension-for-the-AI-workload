#!/usr/bin/env bash
if [ $CLEAN_LOCAL_SOURCES = "1" ]; then
    rm -rf $1/src
    rm -rf $1/verilog_src
    rm -rf $1/cfu.v
fi
