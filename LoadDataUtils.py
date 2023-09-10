import pysam
from Bio import SeqIO
import csv
import numpy as np

denisovans = ["MT576651.1", "NC_013993.1","KT780370.1", "FR695060.1","MT576652.1","MT576653.1"]
neanderthals = ["MG025538.1", "KJ533545.1", "AM948965.1", "KU131206.2","KC879692.1","FM865411.1"]
sapienses = ["AY195749.2","AY195757.1", "AY195781.2","AF346981.1",  "AY195760.2","AY882416.1","AY963586.3"] 

def getListOfReadsFromBamFile(bamFilePath):
    samfile = pysam.AlignmentFile(bamFilePath, "rb")
    list_to_return = []
    for (ind, entry) in enumerate(samfile):
        list_to_return.append({
            "read": str(entry.seq),
            "isReverse": entry.is_reverse,
            "rg":[a[1] for a in entry.get_tags() if a[0] == "RG"][0],
            "index": ind
        })
    return list_to_return

def getListOfReadsFromFastaFile(fastaFilePath):
    ref = SeqIO.parse(fastaFilePath, "fasta")
    refs = [str(i.seq).upper() for i in ref]
    return refs

def create_fasta_from_list_of_reads(fasta_file_name, reads):
    fasta_to_write = open(fasta_file_name, 'w+')
    for s in reads:
        read = f">name | name \n{s}\n"
        fasta_to_write.write(read)
    fasta_to_write.close()

def getKallistoAbundance():
    
    tsv_file = open("output/abundance.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    data = []
    for row in read_tsv:
        data.append(row)
    data = data[1:]
    neanderthals_abundance  = 0
    denisovans_abundance  = 0
    sapienses_abundance  = 0
    for i in data:
        abundance = float(i[4]) / 1000000
        if (i[0] in neanderthals):
            neanderthals_abundance += abundance
        elif (i[0] in sapienses):
            sapienses_abundance += abundance
        elif (i[0] in denisovans):
            denisovans_abundance += abundance
        else:
            print("error", i[0])
    
    res = np.asarray([sapienses_abundance, neanderthals_abundance, denisovans_abundance])
    return res / sum(res)
     
    
