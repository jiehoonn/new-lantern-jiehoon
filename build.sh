#!/bin/bash
set -e
pip install -r requirements.txt
cd ml && python train.py