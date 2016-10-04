"""Utility Functions

This module provides utility functions such as the path navigation, date&time
formatting, csv reader and etc.
"""

from datetime import datetime
import os
import csv

__author__ = 'yunwang@us.ibm.com (Yun Wang)'


def get_csv_reader(file_obj):
    """Get a line iterator from a CSV file. Check if the CSV file contains a
    header and identify deliminator automatically.

    Args:
        file_obj: a file object of a CSV file

    Returns:
       reader: a reader object which will iterate lines in CSV

    """
    head_data = file_obj.read(2048)
    dialect = csv.Sniffer().sniff(head_data)

    file_obj.seek(0)
    reader = csv.reader(file_obj, dialect)

    if csv.Sniffer().has_header(head_data):
        next(reader, None)

    return reader


def get_total_seconds(time_delta):
    """ Calculate a total seconds of a datetime object obtained as a time delta

    Useful for Python 2.6 or before

    Args:
        time_delta: a datetime object obtained as a time delta

    Returns:
       total_seconds: total seconds of the time delta

    Raise:
        AttributeError: input parameter is not a datetime object
    """

    try:
        total_seconds = (time_delta.microseconds +
                         (time_delta.seconds +
                          time_delta.days * 24 * 3600) * 10**6) / 10**6
    except AttributeError:
        raise AttributeError()

    return total_seconds


def unix_time_millis(date_time):
    """ Convert a given python datetime object to milliseconds

    Args:
        date_time: a datetime object

    Returns:
       milliseconds: calculated milliseconds

    """

    time_to_utc = (date_time - datetime.utcfromtimestamp(0))
    milliseconds = get_total_seconds(time_to_utc) * 1000.0

    return milliseconds


class ChangeDirectory(object):
    """Change directory in a with-statement and restore to previous directory
    after exit the statement.
    """
    def __init__(self, new_path):
        """Constructor

        Args:
            new_path: new directory for the nested codes
        """
        self.new_path = os.path.expanduser(new_path)

    def __enter__(self):
        """Save current path
        """
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, evalue, traceback):
        """Restore saved path
        """
        if etype is None and self.saved_path:
            os.chdir(self.saved_path)
        else:
            raise etype, evalue, traceback