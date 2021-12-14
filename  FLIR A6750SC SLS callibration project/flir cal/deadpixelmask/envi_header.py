#!/usr/bin/env python
# -*- coding: utf-8 -*-

###########################################################
# This file has been created by ARSF Data Analysis Node and
# is licensed under the GPL v3 Licence. A copy of this
# licence is available to download with this file.
###########################################################

"""
    @author: Eon Rehman
"""
from __future__ import print_function
import collections
import os
import re
import sys
import numpy

ENVI_TO_NUMPY_DTYPE = {'1':  numpy.uint8,
                       '2':  numpy.int16,
                       '3':  numpy.int32,
                       '4':  numpy.float32,
                       '5':  numpy.float64,
                       '6':  numpy.complex64,
                       '9':  numpy.complex128,
                       '12': numpy.uint16,
                       '13': numpy.uint32,
                       '14': numpy.int64,
                       '15': numpy.uint64}

def find_hdr_file(rawfilename):
    """
    Find ENVI header file associated with data file
    """
    if not os.path.isfile(rawfilename):
        raise IOError("Could not find file " + rawfilename)

    # Get the filename without path or extension
    filename = os.path.basename(rawfilename)
    filesplit = os.path.splitext(filename)
    filebase = filesplit[0]
    dirname = os.path.dirname(rawfilename)

    # See if we can find the header file to use
    if os.path.isfile(os.path.join(dirname, filebase + ".hdr")):
        hdrfile = os.path.join(dirname, filebase + ".hdr")
    elif os.path.isfile(os.path.join(dirname, filename + ".hdr")):
        hdrfile = os.path.join(dirname, filename + ".hdr")
    else:
        hdrfile = None

    return hdrfile

def read_hdr_file(hdrfilename):
    """
    Read information from ENVI header file to a dictionary.
    """
    output = collections.OrderedDict()
    comments = ""
    inblock = False

    try:
        hdrfile = open(hdrfilename, "r")
    except:
        raise IOError("Could not open hdr file " + str(hdrfilename) + \
                      ". Reason: " + str(sys.exc_info()[1]), sys.exc_info()[2])

    # Read line, split it on equals, strip whitespace from resulting strings
    # and add key/value pair to output
    for currentline in hdrfile:
        # ENVI headers accept blocks bracketed by curly braces - check for these
        if not inblock:
            # Split line on first equals sign
            if re.search("=", currentline) is not None:
                linesplit = re.split("=", currentline, 1)
                # Convert all values to lower case
                key = linesplit[0].strip().lower()
                value = linesplit[1].strip()

                # If value starts with an open brace, it's the start of a block
                # - strip the brace off and read the rest of the block
                if re.match("{", value) is not None:
                    inblock = True
                    value = re.sub("^{", "", value, 1)

                    # If value ends with a close brace it's the end
                    # of the block as well - strip the brace off
                    if re.search("}$", value):
                        inblock = False
                        value = re.sub("}$", "", value, 1)
                value = value.strip()
                output[key] = value
            # Check for a comment
            elif re.search("^;", currentline) is not None:
                comments += currentline
        else:
            # If we're in a block, just read the line, strip whitespace
            # (and any closing brace ending the block) and add the whole thing
            value = currentline.strip()
            if re.search("}$", value):
                inblock = False
                value = re.sub("}$", "", value, 1)
                value = value.strip()
            output[key] = output[key] + value

    hdrfile.close()

    output['_comments'] = comments

    return output

def write_envi_header(filename, header_dict):
    """
    Writes a dictionary to an ENVI header file
    """

    # Open header file for writing
    try:
        hdrfile = open(filename, "w")
    except:
        raise IOError("Could not open hdr file {}. ".format(filename))

    hdrfile.write("ENVI\n")
    for key in header_dict.keys():
        # Check not comments key (will write separately)
        if key != "_comments":
            # If it contains commas likely a list so put in curly braces
            if str(header_dict[key]).count(',') > 0:
                hdrfile.write("{} = {{{}}}\n".format(key, header_dict[key]))
            else:
                # Write key at start of line
                hdrfile.write("{} = {}\n".format(key, header_dict[key]))

    # Write out comments at the end
    hdrfile.write(header_dict['_comments'])
    hdrfile.close()