#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

run_eval() {

    SCRIPT_DIR=$(dirname "$(realpath "$0")")
    cd "$SCRIPT_DIR" || exit 1

    LOG_PATH=log
    mkdir -p $LOG_PATH

    execute_time=$(date "+%Y%m%d%H%M%S")
    log_fname=full-${execute_time}.log
    log=$LOG_PATH/$log_fname

    python3 -u eval.py | tee $log
}

run_eval
