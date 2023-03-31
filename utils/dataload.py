
"""
This script contains functions to interact with AWS S3 data.
Note: You need to have appropriate AWS credentials set up on your machine to load CETI data from S3.
"""

import boto3


def get_s3_client():
    """
    Get the AWS S3 client object.
    :return: The AWS S3 client object.
    """

    # get the AWS S3 client object
    s3_client = boto3.client('s3')

    return s3_client


def get_s3_resource():
    """
    Get the AWS S3 resource object.
    :return: The AWS S3 resource object.
    """
    # get the AWS S3 resource object
    s3_resource = boto3.resource('s3')

    return s3_resource



#### AWS S3 & Data Loading Functions ####
def get_s3_data_iterator(s3_client, bucket_name, directory):
    """
    Get the data iterator for the list objects in AWS S3 bucket.
    :param s3_client: AWS S3 client object.
    :param bucket_name: The bucket name.
    :param directory:  The directory to check for objects.

    :return:  The s3 data results iterator.
    """

    # get list objects paginator
    paginator = s3_client.get_paginator('list_objects')

    # get result iterator to iterate through all files
    results = paginator.paginate(Bucket=bucket_name, Prefix=directory)

    return results


def get_s3_file_object(s3_resource, file_key, bucket_name, subdirectory=None):
    """
    Load the file object from the AWS S3 bucket.
    :param s3_resource: AWS S3 resource object.
    :param file_key: The file key.
    :param bucket_name: The bucket name.
    :param subdirectory: The subdirectory to check for the file.
    :return: The file object.
    """


    # if no subdirectory is specified or the file is in the subdirectory
    if not subdirectory or (subdirectory and subdirectory in file_key) :  # if the file is in te PRH subdirectory

        # get the file object from the bucket
        file_object = s3_resource.Object(bucket_name, file_key).get()['Body'].read()

        return file_object