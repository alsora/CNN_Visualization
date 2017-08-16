import xml.etree.ElementTree as ET
from os import walk, mkdir, remove, stat, listdir
import os
import numpy as np
import caffe


def createSamplesDatastructures(images_dir, annotations_dir, interesting_labels):
    samplesNames = []
    samplesImages = []
    samplesLabels = []

    for root, dirs, files in walk(images_dir):
        for image_name in files:
            name, extension = image_name.split(".")

            samplesNames.append(name)

            imageCompletePath = images_dir + '/' + image_name
            image = caffe.io.load_image(imageCompletePath)
            samplesImages.append(image)

            annotationCompletePath = annotations_dir + '/' + name + '.xml'
            label = readLabelFromAnnotation(annotationCompletePath, interesting_labels)
            samplesLabels.append(label)

    return [samplesNames, samplesImages, samplesLabels]


def normalizeData(featuresVector):

    featureVectorsNormalized = []

    for vec in featuresVector:
        vecNormalized = vec / np.linalg.norm(vec)
        featureVectorsNormalized.append(vecNormalized)

    mean = np.mean(featureVectorsNormalized, axis=0)

    featureVectorsNormalizedCentered = []

    for vec in featureVectorsNormalized:
        vecCentered = vec - mean
        featureVectorsNormalizedCentered.append(vecCentered)

    return featureVectorsNormalizedCentered


def readLabelFromAnnotation(annotationFileName, interesting_labels):
    # Parse the given annotation file and read the label

    tree = ET.parse(annotationFileName)
    root = tree.getroot()
    for obj in root.findall('object'):
        label = obj.find("name").text
        if label in interesting_labels:
            return label
        else:
            return 'unknown'


def loadImageNetFiles(folder, numImages = 100, classes=None):

    samples_names = []
    samples_images = []
    samples_labels = []

    for root, dirs, files in walk(folder):
        
        label = root.split(os.path.sep)[-1]

        if classes and label not in classes:
            continue

        for count, image_name in enumerate(files):

            if count == numImages:
                break

            name, extension = image_name.split(".")

            samples_names.append(name)

            image_complete_path = os.path.join(root, image_name)
            image = caffe.io.load_image(image_complete_path)
            samples_images.append(image)
            samples_labels.append(label)            

    return [samples_names, samples_images, samples_labels]



