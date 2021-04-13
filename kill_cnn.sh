#!/usr/bin/env bash

VENVNAME=cnn
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME