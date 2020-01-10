#!/usr/bin/env python

import pickle


def save_object(obj, filename):

    with open(filename, 'wb') as output:  # Overwrites any existing file.

        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):

    with open(filename, 'rb') as f:

        return pickle.load(f)
