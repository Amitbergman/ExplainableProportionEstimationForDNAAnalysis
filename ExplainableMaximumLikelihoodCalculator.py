from Bio import pairwise2
import numpy as np
from joblib import Parallel, delayed
from Bio.pairwise2 import format_alignment
import os
from termcolor import colored
import matplotlib.pyplot as plt
import pandas as pd
import json
import shap
from IPython.display import display
import random
import GenomeParseUtils

alignment_match_score = 1
alignment_mismatch_score = 0.1
gap_open_score = -1
gap_extend_score = -1

class ExplainableMaximumLikelihoodCalculator:
    #Constructor is getting:
    #list_of_reads = input dataset to analyze
    #paths to reference file names
    #path to the substitution matrix to infer likelihoods
    #number_of_jobs = number of threads that will work parallely
    #species - names of species of which we estimate their proportions
    def __init__(self, list_of_reads, ref_neanderthal_file_names, ref_sapien_file_names, ref_denisovan_file_names, path_to_substitution_matrix, number_of_jobs, species=["Homo Sapiens", "Neanderthal", "Denisovan"]) -> None:
        assert(type(list_of_reads) == list and type(list_of_reads[0]) == str)
        assert(type(ref_sapien_file_names) == list and type(ref_sapien_file_names[0]) == str)
        assert(type(ref_denisovan_file_names) == list and type(ref_denisovan_file_names[0]) == str)
        assert(type(ref_neanderthal_file_names) == list and type(ref_neanderthal_file_names[0]) == str)
        assert(type(path_to_substitution_matrix) == str)
        self.list_species_names = species
        self.list_of_reads = list_of_reads
        self.number_of_reads = len(list_of_reads)
        self.number_of_neanderthal_references = len(ref_neanderthal_file_names)
        self.number_of_sapiens_references = len(ref_sapien_file_names)
        self.number_of_denisovans_references = len(ref_denisovan_file_names)
        self.number_of_references = len(ref_neanderthal_file_names) + len(ref_sapien_file_names) + len(ref_denisovan_file_names)

        self.concatenation_of_all_reads = ""
        for i in self.list_of_reads:
            self.concatenation_of_all_reads += i
        
        self.explainer = None

        self.homo_sapien_references = GenomeParseUtils.getListOfGenomes(ref_sapien_file_names)
        self.neanderthal_references = GenomeParseUtils.getListOfGenomes(ref_neanderthal_file_names)
        self.denisovan_references = GenomeParseUtils.getListOfGenomes(ref_denisovan_file_names)

        #read_to_alignment_score[i][j] is the alignment score of read i to the j'th reference
        self.read_to_alignment_score = np.zeros((self.number_of_reads, self.number_of_references))

        self.first_species_reference_ids = GenomeParseUtils.getListOfGenomeIds(ref_sapien_file_names, self.list_species_names[0])
        self.second_species_reference_ids = GenomeParseUtils.getListOfGenomeIds(ref_neanderthal_file_names, self.list_species_names[1])
        self.third_species_reference_ids = GenomeParseUtils.getListOfGenomeIds(ref_denisovan_file_names, self.list_species_names[2])

        #These are indexes that have too low alignment score, hence we ignore them
        self.exclude_indexes = []
        #This is the estimated probabilities to change from letter X to letter Y in index i of the read
        self.substitution_matrix = GenomeParseUtils.load_substitution_matrix_from_file(path_to_substitution_matrix)
        
        self.probabilities_sapiens = [0 for i in range(self.number_of_reads)]
        self.probabilities_neanderthals = [0 for i in range(self.number_of_reads)]
        self.probabilities_denisovans = [0 for i in range(self.number_of_reads)]

        self.alignments_sapienses = [None for i in range(self.number_of_reads)]
        self.alignments_neanderthals = [None for i in range(self.number_of_reads)]
        self.alignments_denisovans = [None for i in range(self.number_of_reads)]

        #For every read - we save the raw probabilities that we calculated - for example raw_probabilities_neanderthals[i][j] will be the probability of read i to be generated from neanderthal reference j
        self.raw_probabilities_sapienses = [None for i in range(self.number_of_reads)]
        self.raw_probabilities_neanderthals = [None for i in range(self.number_of_reads)]
        self.raw_probabilities_denisovans = [None for i in range(self.number_of_reads)]
        
        print(colored("Loading sequences and calculating alignments to all references, this might take a while. Number of reads: ", "blue"), len(self.list_of_reads))
        
        results_from_threads = Parallel(n_jobs=number_of_jobs, backend='multiprocessing')(delayed(self.calculateProbabilitiesForRead)(i) for i in range(self.number_of_reads))
        for (read_index,
            probability_neanderthal,
            probability_sapien,
            probability_denisovan,
            alignments_neanderthals,
            alignments_sapiens,
            alignments_denisovans,
            probabilities_neanderthals,
            probabilities_sapienses,
            probabilities_denisovans) in results_from_threads:
            self.probabilities_neanderthals[read_index] = probability_neanderthal
            self.probabilities_sapiens[read_index] = probability_sapien
            self.probabilities_denisovans[read_index] = probability_denisovan
            self.alignments_neanderthals[read_index] = json.loads(alignments_neanderthals)
            self.alignments_sapienses[read_index] = json.loads(alignments_sapiens)
            self.alignments_denisovans[read_index] = json.loads(alignments_denisovans)
            self.raw_probabilities_neanderthals[read_index] = probabilities_neanderthals
            self.raw_probabilities_sapienses[read_index] = probabilities_sapienses
            self.raw_probabilities_denisovans[read_index] = probabilities_denisovans

        
        self.normalized_probabilities_vector = [None for i in range(self.number_of_reads)]
        for i in range(self.number_of_reads):
            normalized_probabilities_vector = self.__getNormalizedProbabilitiesVector(i)
            self.normalized_probabilities_vector[i] = normalized_probabilities_vector

        #Fill the matrix of alignment scores with the values
        index_of_score_in_alignment = 2
        for read_index in range(self.number_of_reads):
            alignment_scores = []
            length_of_read = len(self.list_of_reads[read_index])
            for align in self.alignments_sapienses[read_index]:
                alignment_score = align[index_of_score_in_alignment]
                alignment_scores.append(alignment_score/length_of_read)
            for align in self.alignments_neanderthals[read_index]:
                alignment_score = align[index_of_score_in_alignment]
                alignment_scores.append(alignment_score/length_of_read)
            for align in self.alignments_denisovans[read_index]:
                alignment_score = align[index_of_score_in_alignment]
                alignment_scores.append(alignment_score/length_of_read)
            self.read_to_alignment_score[read_index] = alignment_scores
            average_score_for_read = np.average(alignment_scores)
            if (average_score_for_read < 0.85):
                self.exclude_indexes.append(read_index)
                print(colored(f"Excluding read index {read_index} due to too low alignment score: {average_score_for_read}", "red"))

    def getGeneralStatistics(self):
        average_read_length = len(self.concatenation_of_all_reads) / self.number_of_reads
        print(f"Average read length: {average_read_length}")
        
        alleles = dict()
        print("Allele frequencies in the data:")
        all_leters_in_reads = list(set(self.concatenation_of_all_reads))
        for character in all_leters_in_reads:
            alleles[character] = self.concatenation_of_all_reads.count(character)/len(self.concatenation_of_all_reads)
        for character_1 in all_leters_in_reads:
            for character_2 in all_leters_in_reads:
                concat = character_1+character_2
                alleles[concat] = self.concatenation_of_all_reads.count(concat)/len(self.concatenation_of_all_reads)
        data_frame = pd.DataFrame.from_dict(alleles, orient='index', columns=['frequency'])
        display(data_frame)

    def calculateProbabilitiesForRead(self, read_index):

        #We are aligning the sequence to one of the references, to get the general area in which this sequence is located in the reference
        #Then, the other alignment should just search for the best alignment in this area (to make the computing of the alignments faster)
        #This means that we only need to do full alignment once to every sequence
        if (read_index%40 ==0):
            print(f"start working on read number {read_index}")
        initial_alignment = pairwise2.align.localms(self.homo_sapien_references[0], self.list_of_reads[read_index], 
            alignment_match_score,
            alignment_mismatch_score,
            gap_open_score,
            gap_extend_score,
            score_only=False)
        start = initial_alignment[0].start
        end = initial_alignment[0].end

      
        (probability_neanderthal, alignments_neanderthals, probabilities_neanderthals) = self.__probabilityOfReferenceSetToGenerateSequence(read_index, self.neanderthal_references, start, end)
        (probability_sapien, alignments_sapiens, probabilities_sapienses) = self.__probabilityOfReferenceSetToGenerateSequence(read_index, self.homo_sapien_references, start, end)
        (probability_denisovan, alignments_denisovans, probabilities_denisovans) = self.__probabilityOfReferenceSetToGenerateSequence(read_index, self.denisovan_references, start, end)
        
        return (read_index,
                probability_neanderthal,
                probability_sapien,
                probability_denisovan,
                json.dumps(alignments_neanderthals),
                json.dumps(alignments_sapiens),
                json.dumps(alignments_denisovans),
                probabilities_neanderthals,
                probabilities_sapienses,
                probabilities_denisovans
                )

    def __probabilityThatAWillGenerateB(self,a,b):
        assert(len(a) == len(b))
        probability_to_return = 1
        for i in range(len(a)):
            from_a = a[i]
            from_b = b[i]

            index_in_change_frequency = -1
            if (i < 10):
                index_in_change_frequency = i
            elif (i > len(a)-10):
                index_in_change_frequency = (len(a) - i-1) + 15 #Index from the end
            else:
                index_in_change_frequency = 30
            
            if (from_a != 'N' and from_b != 'N' and from_a != "-" and from_b != '-'):
                probability_of_the_change = self.substitution_matrix[index_in_change_frequency][from_a+from_b]
            else:
                probability_of_the_change = 0.01
            probability_to_return = probability_to_return * probability_of_the_change
        return probability_to_return
    
    def __probabilityOfReferenceSetToGenerateSequence(self, sequence_index, references_list, start, end):
        sequence = self.list_of_reads[sequence_index]
        probabilities = []
        alignments = []
        for reference in references_list:
            #We do not need to look at all the reference, we already know where in general the sequence is going to be aligned to
            start_in_reference = max([start-100, 0])
            end_in_reference = min([end+100, len(reference)])
            alignment = pairwise2.align.localms(
                reference[start_in_reference:end_in_reference],
                sequence,
                alignment_match_score,
                alignment_mismatch_score,
                gap_open_score,
                gap_extend_score,
                score_only=False)[0]
            alignments.append(alignment)
            alignment_start = alignment.start
            alignment_end = alignment.end
            seq_a = alignment.seqA[alignment_start:alignment_end]
            seq_b = alignment.seqB[alignment_start:alignment_end]
            probability = self.__probabilityThatAWillGenerateB(seq_a, seq_b)
            probabilities.append(probability)
        average_probability = np.average(probabilities)
        return (average_probability, alignments, probabilities)

    #First param is the alpha of the sapiens, second is the alpha of the neanderthal, third is a denisovan
    def calc_likelihood(self, alphas_vector, ignore_read_indexes=[]) -> float:
       
        if (sum(alphas_vector)>1.00001 or sum(alphas_vector) < 0.9999):
            raise Exception(f"Sum of alphas must be 1. Got the sum: {sum(alphas_vector)}")
        if (min(alphas_vector) < -0.000001):
            raise Exception(f"Alpha must be positive value. You entered a value of {min(alphas_vector)}")
        likelihood_of_data = 1.0
        indexes_to_cover = [i for i in range(self.number_of_reads) if i not in ignore_read_indexes and i not in self.exclude_indexes]
        for i in indexes_to_cover:
            normalized_probabilities_vector = self.normalized_probabilities_vector[i]
            likelihood_of_current_read = np.dot(alphas_vector, normalized_probabilities_vector)
            likelihood_of_data = likelihood_of_data * likelihood_of_current_read
        return likelihood_of_data
    
    def get_A_s_d_values(self):
        return self.normalized_probabilities_vector

    def calc_likelihood_on_partial_references(self, alphas, subset_of_read_indexes, probabilities_of_partial_references):
        
        likelihood_of_data = 1.0
        for i in subset_of_read_indexes:
            likelihood_of_current_read = np.dot(alphas, probabilities_of_partial_references[i])
            likelihood_of_data = likelihood_of_data * likelihood_of_current_read
        return likelihood_of_data

    def __getNormalizedProbabilitiesVector(self, index):
        probability_sapien = self.probabilities_sapiens[index]
        probability_neanderthal = self.probabilities_neanderthals[index]
        probability_denisovan = self.probabilities_denisovans[index]
        #We do not want that reads with very low probability will be ignored by the process
        #So we normalize the probabilities
        #TODO: Understand if this is the right way to go
        return self.__normalizeAVector([probability_sapien, probability_neanderthal, probability_denisovan])

    def __normalizeAVector(self, vector):
        probabilities_sum = sum(vector)
        normalized_probability_sapien = vector[0]/ probabilities_sum
        normalized_probability_neanderthal = vector[1] / probabilities_sum
        normalized_probability_denisovan = vector[2] / probabilities_sum
        return [normalized_probability_sapien, normalized_probability_neanderthal, normalized_probability_denisovan]

    def calc_maximum_likelihood_on_subset(self, subset_of_indexes, result_resolution=40):
        array = np.zeros((1,3))
        features = ["Homo Sapiens", "Neanderthals", "Denisovans"]
        rows = ["Result"]
        if (len(subset_of_indexes)==0):
            # Don't know anything, just guess uniform over them
            array[0][0] = 0.33
            array[0][1] = 0.33
            array[0][2] = 0.34
            return pd.DataFrame(array, columns=features, index=rows)
        if ((min(subset_of_indexes) < 0) or max(subset_of_indexes) > self.number_of_reads):
            raise Exception("error - subset of indexes is problematic")
        ignore_indexes = [i for i in range(len(self.list_of_reads)) if i not in subset_of_indexes]
        return self.estimate_species_proportions(ignore_list_indexes=ignore_indexes, result_resolution=result_resolution)
    
    def calc_all_likelihoods(self):
        results = []
        accuracy = 40
        for i in range (accuracy+1):
            for j in range(accuracy+1):
                a_1 = i/accuracy 
                a_2 = j/accuracy
                a_3 = 1-(a_2 + a_1)
                if (a_3 >=0):
                    res = self.calc_likelihood([a_1, a_2, a_3])
                    results.append((a_1,a_2, a_3,res))
        array = np.asarray(results)
        return pd.DataFrame(array, columns=["Sapiens alpha", "Neanderthal alpha", "Denisovan alpha", "likelihood"]).sort_values("likelihood", ascending=False, ignore_index=True)

    #Result_resolution is how accurate we want to be, if result resolution is 10, the response will be
    #In multiplication of 0.1
    #If result resolution is 20, the result will be in mulitplications of 0.05
    #If result resolution is 100, the result will be in multiplications of 0.01
    #The higher it is, the slower the algorithm, but it is more accurate
    def estimate_species_proportions(self, result_resolution=40,ignore_list_indexes=[]):
        array = np.zeros((1,3))
        features = ["Homo Sapiens", "Neanderthals", "Denisovans"]
        rows = ["Estimation"]
        assert(len(ignore_list_indexes) <= len(self.list_of_reads))
        if (len(ignore_list_indexes) == len(self.list_of_reads)):
            # Don't know anything, just guess uniform over them
            array[0][0] = 0.33
            array[0][1] = 0.33
            array[0][2] = 0.34
            return pd.DataFrame(array, index=rows, columns=features)
        max_likelihood = 0
        max_alphas = (-1,-1,-1)
        for i in range (result_resolution+1):
            for j in range(result_resolution+1):
                a_1 = i/result_resolution 
                a_2 = j/result_resolution
                a_3 = 1-(a_2 + a_1)
                if (a_3 >=0):
                    likelihood = self.calc_likelihood([a_1, a_2, a_3], ignore_list_indexes)
                    if (likelihood > max_likelihood):
                        max_likelihood = likelihood
                        max_alphas = (a_1,a_2, a_3,likelihood)
        array[0][0] = max_alphas[0]
        array[0][1] = max_alphas[1]
        array[0][2] = max_alphas[2]
        return pd.DataFrame(array, index=rows, columns=features)
    
    def calc_maximum_likelihoods_3_references_on_partial_reference_sets(self, result_resolution=40,reference_index_sapien = [], reference_index_neanderthal = [], reference_indexes_denisovan = [], sample=None):
        result = np.zeros((1,3))
        if (sample == None):
            sample = [i for i in range(self.number_of_reads)]
        max_likelihood = 0
        max_alphas = (-1,-1,-1)
        probabilities_for_likelihood =  np.zeros((self.number_of_reads, 3))
        for i in range(self.number_of_reads):
            relevent_probabilities_sapienses = [self.raw_probabilities_sapienses[i][k] for k in reference_index_sapien]
            relevent_probabilities_neanderthals = [self.raw_probabilities_neanderthals[i][k] for k in reference_index_neanderthal]
            relevent_probabilities_denisovans = [self.raw_probabilities_denisovans[i][k] for k in reference_indexes_denisovan]
            average_sapienses = np.mean(relevent_probabilities_sapienses)
            average_neanderthals = np.mean(relevent_probabilities_neanderthals)
            average_denisovans = np.mean(relevent_probabilities_denisovans)
            probabilities_for_likelihood[i] = self.__normalizeAVector([average_sapienses, average_neanderthals, average_denisovans])
        
        for i in range (result_resolution+1):
            for j in range(result_resolution+1):
                a_1 = i/result_resolution 
                a_2 = j/result_resolution
                a_3 = 1-(a_2 + a_1)
                if (a_3 >=0):
                    likelihood = self.calc_likelihood_on_partial_references(
                        [a_1, a_2, a_3],
                        sample,
                        probabilities_for_likelihood)
                    if (likelihood > max_likelihood):
                        max_likelihood = likelihood
                        max_alphas = (a_1,a_2, a_3,likelihood)
        result[0][0] = max_alphas[0]
        result[0][1] = max_alphas[1]
        result[0][2] = max_alphas[2]
        return result

    def estimate_shapley_value_for_read(self, read_index, number_of_samples_per_read):
        print(colored(f"Start working on read number {read_index} in processId {os.getpid()}", "green"))
        possible_indexes = [i for i in range(self.number_of_reads) if i!= read_index]
        results_with_minus_without = []
        results_with_minus_without_not_scaled = []
        for i in range(number_of_samples_per_read):
            sample_size =  random.randint(1,len(possible_indexes))
            sample_without = random.sample(possible_indexes, sample_size)
            sample_with = sample_without.copy()
            sample_with.append(read_index)

            model_result_with_index = np.asarray(self.calc_maximum_likelihood_on_subset(sample_with, result_resolution=50).values.flatten())
            model_result_without_index = np.asarray(self.calc_maximum_likelihood_on_subset(sample_without, result_resolution=50).values.flatten())
            current_diff = model_result_with_index - model_result_without_index
            scaled_diff = current_diff * sample_size
            results_with_minus_without.append(scaled_diff)
            results_with_minus_without_not_scaled.append(current_diff)
        average_influence_scaled = np.mean(np.asarray(results_with_minus_without), axis=0)
        average_influence_not_scaled = np.mean(np.asarray(results_with_minus_without_not_scaled), axis=0)
        return (read_index, average_influence_scaled, average_influence_not_scaled)


    def estimate_shapley_values(self, number_of_samples_per_read=200, number_of_jobs=-1):
        #This will return for every species the influence of every read on the value of the model
        #For example, results[0][i] will be the influence of read i on "Sapiens" value of the result 
        #For example, results[1][i] will be the influence of read i on "Neanderthals" value of the result 
        #For example, results[2][i] will be the influence of read i on "Denisovans" value of the result 
        #Positive number in sapiens influece means that this read made the model lean more to the direction of saying "Sapiens" 
        
        
        sample_to_run = [i for i in range(self.number_of_reads)]
        averageInfluece = np.zeros((self.number_of_reads, 3))

        change_dict = dict()
        for i in sample_to_run:
            change_dict[i] = []
        for i in range(number_of_samples_per_read):
            permutation = GenomeParseUtils.get_random_perm(sample_to_run)
            for length in range(len(permutation)):
                current_data_with = permutation[:length+1]
                current_data_without = permutation[:length]
                item_we_are_adding = permutation[length]
                model_result_with_index = np.asarray(self.calc_maximum_likelihood_on_subset(current_data_with, result_resolution=50).values.flatten())
                model_result_without_index = np.asarray(self.calc_maximum_likelihood_on_subset(current_data_without, result_resolution=50).values.flatten())
                current_diff = model_result_with_index - model_result_without_index
                change_dict[item_we_are_adding].append(current_diff)

        for i in sample_to_run:
            averageInfluece[i] = GenomeParseUtils.get_average_of_list(change_dict[i])

        sapienses_shapley_estimation_not_scaled = averageInfluece[None,:,0]
        neanderthals_shapley_estimation_not_scaled = averageInfluece[None,:,1]
        denisovans_shapley_estimation_not_scaled = averageInfluece[None,:,2]
        results = [sapienses_shapley_estimation_not_scaled,neanderthals_shapley_estimation_not_scaled, denisovans_shapley_estimation_not_scaled]
        return results
        
    def plot_influence_values(self, influence_values):
        sapienses = influence_values[0].flatten()
        neanderthals = influence_values[1].flatten()
        denisovans = influence_values[2].flatten()
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        reads = [i for i in range(self.number_of_reads)]
        ax.bar(reads,sapienses)
        ax.set_xlabel("read index")
        ax.set_ylabel("influence on results - Homo Sapiens")
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(reads,neanderthals)
        ax.set_xlabel("read index")
        ax.set_ylabel("influence on results - Neanderthals")
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(reads,denisovans)
        ax.set_xlabel("read index")
        ax.set_ylabel("influence on results - Denisovans")
        plt.show()

    def plot_alignments_of_read(self, read_index):
        read_length = len(self.list_of_reads[read_index])
        print(colored(f"Printing all alignments for read number {read_index}:", "green"))
        print(colored(f"Read length is {read_length}:", "green"))
        print(colored("Alignments to Homo sapiens references:", "blue"))
        for i in range(len(self.alignments_sapienses[read_index])):
            print(self.first_species_reference_ids[i])
            print(format_alignment(*self.alignments_sapienses[read_index][i]))  
        print(colored("Alignments to Neanderthal references:", "blue"))
        for i in range(len(self.alignments_neanderthals[read_index])):
            print(self.second_species_reference_ids[i])
            print(format_alignment(*self.alignments_neanderthals[read_index][i]))  
        print(colored("Alignments to Denisovan references:", "blue"))
        for i in range(len(self.alignments_denisovans[read_index])):
            print(self.third_species_reference_ids[i])
            print(format_alignment(*self.alignments_denisovans[read_index][i]))   

    def getReadToAlignmentScoreTable(self):
        data_frame = pd.DataFrame(self.read_to_alignment_score, columns=[self.first_species_reference_ids + self.second_species_reference_ids + self.third_species_reference_ids])
        data_frame.loc['mean'] = data_frame.mean()
        return data_frame
    
    def getReadToAlignmentScoreMeanOfSubsetOfData(self, subset_to_include):

        mean = self.read_to_alignment_score[subset_to_include].mean(axis=0).reshape(1,self.number_of_references)
        data_frame = pd.DataFrame(mean, columns=[self.first_species_reference_ids + self.second_species_reference_ids + self.third_species_reference_ids])

        return data_frame

    def getMaximumLikelihoodOnSample(self, samples,  index, number_of_features):
        indexes = [i for i in range(number_of_features) if samples[index][i] ==1]
        result_on_indexes = self.calc_maximum_likelihood_on_subset(indexes, 50).values.flatten()
        return (index, result_on_indexes)

    def calculateModelResultsOnSamples(self, matrix_of_samples):
        number_of_samples = matrix_of_samples.shape[0]
        number_of_features = matrix_of_samples.shape[1]
        matrix_to_return = np.zeros((number_of_samples,3))
        print(number_of_samples)
        #Paralelly calculate the model on the samples
        results_from_threads = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(self.getMaximumLikelihoodOnSample)(matrix_of_samples, i, number_of_features) for i in range(len(matrix_of_samples)))
        for (index, result) in results_from_threads:
            matrix_to_return[index] = result
        return matrix_to_return
    
    #This treats every read in the sample as a feature, and finds the shap values of them
    #Meaning what is the contribution of each of the reads to the final result
    def calculate_shapley_values(self, nsamples=10000):
        
        all_zeros_sample = np.zeros((1,self.number_of_reads))
        all_ones_sample = np.ones((1,self.number_of_reads))
        explainer = shap.KernelExplainer(self.calculateModelResultsOnSamples, all_zeros_sample)
        self.explainer = explainer
        shap_values = explainer.shap_values(all_ones_sample, nsamples=nsamples)
        return shap_values
    

    def getRankingOfA_d_s_Values(self):
        a_d_s = self.get_A_s_d_values()
        order = []
        for label in range(len(self.list_species_names)):
            relevant_to_label = list(enumerate([i[label] for i in a_d_s]))
            sorted_list = sorted(relevant_to_label, key=lambda a: a[1])
            only_ind = [i[0] for i in sorted_list]
            order.append((self.list_species_names[label], only_ind))
        return order

    def plot_shap_values(self, shap_values):

        print("summary plot:")
        sample = np.ones((1,self.number_of_reads))
        shap.initjs()
        shap.summary_plot(shap_values=shap_values,
                features=sample,
                plot_type="bar",
                class_names = ["Sapien", "Neanderthal", "Denisovan"],
                feature_names = ["Read " + str(i) for i in range(self.number_of_reads)],
                max_display=40,
                )

        print("Force plots")
        print("Homo Sapiens:")
        shap.force_plot(self.explainer.expected_value[0], shap_values[0], feature_names=["Read " + str(i) for i in range(self.number_of_reads)], matplotlib=True)

        print("Neanderthals:")
        shap.force_plot(self.explainer.expected_value[1], shap_values[1], feature_names=["Read " + str(i) for i in range(self.number_of_reads)], matplotlib=True)

        print("Denisovans:")
        shap.force_plot(self.explainer.expected_value[2], shap_values[2], feature_names=["Read " + str(i) for i in range(self.number_of_reads)], matplotlib=True)


        print("Decision plots:")
        print("Homo Sapiens")
        shap.decision_plot(self.explainer.expected_value[0], shap_values[0], feature_names=["Read " + str(i) for i in range(self.number_of_reads)])

        print("Neanderthals:")
        shap.decision_plot(self.explainer.expected_value[1], shap_values[1], feature_names=["Read " + str(i) for i in range(self.number_of_reads)])

        print("Denisovans:")
        shap.decision_plot(self.explainer.expected_value[2], shap_values[2], feature_names=["Read " + str(i) for i in range(self.number_of_reads)])

        print("Multi output decision plot:")
        shap.multioutput_decision_plot(list(self.explainer.expected_value), shap_values, row_index=0)
    
    def getMostInfluencingReads(self, shap_values, reads_to_print):
        sapienses = shap_values[0][0]
        neanderthals = shap_values[1][0]
        denisovans = shap_values[2][0]

        sapienses_with_index = [(i, sapienses[i]) for i in range(len(sapienses))]
        neanderthals_with_index = [(i, neanderthals[i]) for i in range(len(neanderthals))]
        denisovans_with_index = [(i, denisovans[i]) for i in range(len(denisovans))]

        print("Most influencing reads for sapiens (and impact):")
        sapienses = sorted(sapienses_with_index, key=lambda a: np.absolute(a[1]), reverse=True)[:reads_to_print]
        print(sapienses)
       
        print("Most influencing reads for neanderthals (and impact):")
        neanderthals = sorted(neanderthals_with_index, key=lambda a: np.absolute(a[1]), reverse=True)[:reads_to_print]
        print(neanderthals)

        print("Most influencing reads for denisovans (and impact):")
        denisovans = sorted(denisovans_with_index, key=lambda a: np.absolute(a[1]), reverse=True)[:reads_to_print]
        print(denisovans)
        return [sapienses, neanderthals, denisovans]

    def analyze_diff_on_removing_reference(self, number_of_samples=20, size_of_sample = 10, result_resolution=50):
        indexes = [i for i in range(self.number_of_reads)]
        if (size_of_sample >= self.number_of_reads):
            size_of_sample = self.number_of_reads // 2
        data = []
        number_of_neanderthals = len(self.second_species_reference_ids)
        number_of_sapienses = len(self.first_species_reference_ids)
        number_of_denisovans = len(self.third_species_reference_ids)
        for i in range(number_of_samples):
            sample = list(np.random.choice(indexes, size_of_sample, replace=False))
            result_with_all_references = self.calc_maximum_likelihood_on_subset(sample, result_resolution=result_resolution).values[0]
            for neanderthal_to_remove in range(number_of_neanderthals):
                neanderthal_refs_to_include = [i for i in range(number_of_neanderthals) if i != neanderthal_to_remove]
                sapiens_refs_to_include = [i for i in range(number_of_sapienses)]
                denisovan_refs_to_include = [i for i in range(number_of_denisovans)]
                
                without_reference_i = self.calc_maximum_likelihoods_3_references_on_partial_reference_sets(result_resolution,sapiens_refs_to_include, neanderthal_refs_to_include, denisovan_refs_to_include, sample)[0]
                diff =  result_with_all_references - without_reference_i
                #The diff is the influence of adding reference i to the result
                data.append((neanderthal_to_remove, diff))
            # Need to execute the same also for denisovan and Homo Sapiens references
        return data

    def draw_only_important_features(self, shap_values, species_index, number_of_features_to_draw=5):
        t = shap_values[species_index][0]
        with_indexes = list(enumerate(t))
        t = sorted(with_indexes, key=lambda y: np.absolute(y[1]))[:number_of_features_to_draw]
        values = [i[1] for i in t]
        names = ["read " + str(i[0]) for i in t]
        shap.force_plot(self.explainer.expected_value[0], np.asarray(values), feature_names=names, matplotlib=True)


    def influenceOfReference(self, reference_index_to_remove, result_resolution=40):
        modified_probabilities_without_reference_i = np.zeros((self.number_of_reads, 3))
        modified_probabilities_without_reference_i[:,0] = self.probabilities_sapiens
        modified_probabilities_without_reference_i[:,2] = self.probabilities_denisovans
        
        for read_ind in range(self.number_of_reads):
            probability_of_read = (self.probabilities_neanderthals[read_ind] * self.number_of_neanderthal_references - self.raw_probabilities_neanderthals[read_ind][reference_index_to_remove])/(self.number_of_neanderthal_references - 1)
            modified_probabilities_without_reference_i[read_ind][1] = probability_of_read
        
        number_of_samples = 50
        all_indexes = [i for i in range(self.number_of_reads)]
        data = []
        for s in range(number_of_samples):
            sample = list(np.random.choice(all_indexes, 10, replace=False)) 
            not_in_sample = [i for i in all_indexes if i not in sample]

            max_likelihood_withoutref = 0
            result_without_reference = np.asarray([-1,-1,-1])

            max_likelihood_with_all_refs = 0
            result_with_all_references = np.asarray([-1,-1,-1])
            for i in range (result_resolution+1):
                for j in range(result_resolution+1):
                    a_1 = i/result_resolution 
                    a_2 = j/result_resolution
                    a_3 = 1-(a_2 + a_1)
                    if (a_3 >=0):
                        likelihood_without_reference = self.calc_likelihood_on_partial_references(
                            [a_1, a_2, a_3],
                            sample,
                            modified_probabilities_without_reference_i)
                        if (likelihood_without_reference > max_likelihood_withoutref):
                            max_likelihood_withoutref = likelihood_without_reference
                            result_without_reference = np.asarray([a_1, a_2, a_3])
                        likelihood_with = self.calc_likelihood([a_1, a_2, a_3], not_in_sample)
                        if (likelihood_with > max_likelihood_with_all_refs):
                            max_likelihood_with_all_refs = likelihood_with
                            result_with_all_references = np.asarray([a_1, a_2, a_3])
            with_minus_without = result_with_all_references - result_without_reference
            data.append(with_minus_without)
        #return the average influence of removing this reference        
        return np.mean(data, 0)

    #Generate counter factual 1, given shapley influence values                
    def generateCounterFactualMinimalSetToRemoveAndChangeMax(self, influence_values):
        #current_maximizer = self.max_3_references()
        #calculate reads that are most influential against it
        #remove them one by one and while the maximum remains the same
        current_max = self.estimate_species_proportions().values.argmax()
        influence = influence_values[current_max][0]
        org = [(i, influence[i]) for i in range(self.number_of_reads)]
        s = sorted(org, key=lambda a:a[1], reverse=True)
        current_reads_to_ignore = []
        i=0
        success = False
        while(len(current_reads_to_ignore) < self.number_of_reads-1):
            current_reads_to_ignore.append(s[i][0])
            i+=1
            max_likelihood = self.estimate_species_proportions(ignore_list_indexes=current_reads_to_ignore)
            max_after = max_likelihood.values.argmax()
            if (max_after != current_max):
                success = True
                break
        
        if (success):
            print(f"Change dominating species from {self.list_species_names[current_max]} to {self.list_species_names[max_after]} would require removing {len(current_reads_to_ignore)} reads: {current_reads_to_ignore}")
            return (current_reads_to_ignore, max_after)
        else:
            return ([i for i in range(self.number_of_reads)], "")
        
    #Generate counter factual 1, using the a_s_d values    
    def generateCounterFactualMinimalSetToRemoveAndChangeMax_using_a_s_d_values(self):
        #current_maximizer = self.max_3_references()
        #calculate reads that are most influential against it
        #remove them one by one and while the maximum remains the same
        current_max = self.estimate_species_proportions().values.argmax()
        a_s_d_values = self.get_A_s_d_values()
        influence_on_highest = [t[current_max] for t in a_s_d_values]
        org = [(i, influence_on_highest[i]) for i in range(self.number_of_reads)]
        s = sorted(org, key=lambda a:a[1], reverse=True)
        current_reads_to_ignore = []
        i=0
        success = False
        while(len(current_reads_to_ignore) < self.number_of_reads-1):
            current_reads_to_ignore.append(s[i][0])
            i+=1
            max_likelihood = self.estimate_species_proportions(ignore_list_indexes=current_reads_to_ignore)
            max_after = max_likelihood.values.argmax()
            if (max_after != current_max):
                success = True
                break
        
        if (success):
            print(f"Change dominating species from {self.list_species_names[current_max]} to {self.list_species_names[max_after]} would require removing {len(current_reads_to_ignore)} reads: {current_reads_to_ignore}")
            return (current_reads_to_ignore, max_after)
        else:
            return ([], "")

    
    #Optimize the likelihood function usinge a variant of the gradient descent algorithm
    def estimate_species_proportions_gradient_descent(self, number_of_starting_points = 1, number_of_iterations=100, change_rate = 0.01, ignore_list_indexes=[]):
        assert(len(ignore_list_indexes) < len(self.list_of_reads))
        array = np.zeros((1,3))
        features = ["Homo Sapiens", "Neanderthals", "Denisovans"]
        rows = ["Estimation"]
        current_all = np.asarray([0.3,0.3,0.4])
        baseline = self.calc_likelihood(current_all, ignore_list_indexes)
        best_value_all = baseline
        for i in range(number_of_starting_points):
            a = random.randint(0,10)
            b = random.randint(0,10-a)
            c = 10-a-b
            starting_point = np.asarray([a/10,b/10,c/10])
            current = starting_point
            
            for i in range(number_of_iterations):
                option_a = self.__makeSumOneAndVerifyNoNegative__(current + [change_rate, -change_rate, 0])
                option_b = self.__makeSumOneAndVerifyNoNegative__(current + [-change_rate, change_rate, 0])
                option_c = self.__makeSumOneAndVerifyNoNegative__(current + [change_rate,0, -change_rate])
                option_d = self.__makeSumOneAndVerifyNoNegative__(current + [-change_rate,0, change_rate])
                option_e = self.__makeSumOneAndVerifyNoNegative__(current + [0,change_rate, -change_rate])
                option_f = self.__makeSumOneAndVerifyNoNegative__(current + [0, -change_rate, change_rate])
                options = [option_a, option_b, option_c, option_d, option_e, option_f, current]
                options_with_results = [(a,self.calc_likelihood(a, ignore_list_indexes)) for a in options]
                best = max(range(len(options_with_results)), key=lambda ind: options_with_results[ind][1])
                current = options[best]
                best_value_current = options_with_results[best][1]
            if (best_value_current > best_value_all):
                best_value_all = best_value_current
                current_all = current
        array[0][0] = current_all[0]
        array[0][1] = current_all[1]
        array[0][2] = current_all[2]
        return pd.DataFrame(array, index=rows, columns=features)
    
    def __makeSumOneAndVerifyNoNegative__(self, vector):
        if min(vector) < 0:
            return np.asarray([1,0,0])
        return np.asarray([vector[0], vector[1], 1-(vector[0]+ vector[1])])