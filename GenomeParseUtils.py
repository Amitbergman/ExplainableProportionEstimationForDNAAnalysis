from Bio import SeqIO
import csv
import random
import numpy as np

def getListOfGenomes(fileNames):
    genomes = []
    for fileName in fileNames:
        genome = getGenome(fileName)
        genomes.append(genome)
    return genomes

    
def getGenome(fileName):
    genome_as_fasta = list(SeqIO.parse(fileName, "fasta"))[0]
    genome = str(genome_as_fasta.seq)
    circular_genome = makeGenomeCircular(genome)
    return circular_genome

def makeGenomeCircular(genome, max_read_length=100):
    return genome + genome[:max_read_length]

def get_average_of_list(list_of_influences):
    average_influence = np.mean(np.asarray(list_of_influences), axis=0)
    return average_influence

def get_random_perm(group_elements):
    random_permutation = random.sample(group_elements, len(group_elements))
    return random_permutation


def getListOfGenomeIds(fileNames, specieNameToAdd):
        genomeIds = []
        for fileName in fileNames:
            genomeid = getGenomeId(fileName)
            genomeIds.append(specieNameToAdd + "-" + genomeid)
        return genomeIds
    
def getGenomeId(fileName):
        genome_as_fasta = list(SeqIO.parse(fileName, "fasta"))[0]
        return genome_as_fasta.id


def load_substitution_matrix_from_file(path):
    tsv_file = open(path)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    list_of_frequencies = []
    list_of_dictionaries = [dict() for i in range(31)]
    for row in read_tsv:
        list_of_frequencies.append(row)
    for row in list_of_frequencies:
        index = row[0]
        from_letter = row[1]
        to_letter = row[2]
        frequency = row[3]
        from_to = from_letter + to_letter
        list_of_dictionaries[int(str(index))][from_to] = float(frequency)
    return list_of_dictionaries