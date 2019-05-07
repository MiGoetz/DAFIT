"""
Copyright (c) German Cancer Research Center,
Division of Medical Image Computing.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Author: Michael Goetz
Mail: m.goetz@dkfz-heidelberg.de
 
"""
import numpy.random as rand
import numpy as np


class TwoClassProblem:
    means_A=None
    stddev_A=None
    means_B=None
    stddev_B=None
    means_C=None
    stddev_C=None
    samples_A=None
    samples_B=None
    samples=None
    stddev_noise=None
    noise_samples_A=None
    noise_samples_B=None
    noise_samples=None
    noise_labels=None
    
    validation_x=None
    validation_y=None

"""
Can be used to generate two vectors that describe a feature distribution 
by defining the mean and standard deviation along each dimension. 

Returns: means, stddevs. 
        Means is a vector of mean values, and stddevs a vector of standard deviations
"""
def generate_feature_matrix_uniform(dimensions, mean_range=[-2,2], stddev_range=[0.1,1], seed=None):
    if len(mean_range) != 2:
        raise Exception("Expected a two dimensional range for mean values. Got " + str(mean_range))
    if len(stddev_range) != 2: 
        raise Exception("Expected a two dimensional range for std values. Got " + str(stddev_range))
    if np.min(stddev_range) <= 0:
        raise Exception("Range for Std.Dev. must be positive. Got " + str(stddev_range))
    if seed is not None:
        rand.seed(seed)
    means = rand.uniform(mean_range[0], mean_range[1], dimensions)
    stddevs = rand.uniform(stddev_range[0], stddev_range[1], dimensions)
    return means, stddevs

def draw_samples_from_distribution(means, stddevs, samples):
    dimension=len(means)
    draws=rand.normal(means, stddevs, (samples,dimension))
    return draws

def add_gaussian_noise_to_samples(samples, means, stddevs):
    if means is None:
        means=np.zeros(samples.shape[1])
    noise=draw_samples_from_distribution(means, stddevs, len(samples))
    return samples+noise

def generate_two_class_problem(number_of_samples, dimensions, mean_range_A, mean_range_B, stddev_range_A, stddev_range_B, stddev_noise_range, seed=None):
    erg=TwoClassProblem()
    erg.means_A, erg.stddev_A = generate_feature_matrix_uniform(dimensions, mean_range_A, stddev_range_A, seed)
    erg.means_B, erg.stddev_B = generate_feature_matrix_uniform(dimensions, mean_range_B, stddev_range_B)
    erg.samples_A=draw_samples_from_distribution(erg.means_A, erg.stddev_A, number_of_samples)
    erg.samples_B=draw_samples_from_distribution(erg.means_B, erg.stddev_B, number_of_samples)
    means_noise, erg.stddev_noise=generate_feature_matrix_uniform(dimensions, mean_range_A, stddev_noise_range)
    erg.noise_samples_A=add_gaussian_noise_to_samples(erg.samples_A, None, erg.stddev_noise)
    erg.noise_samples_B=add_gaussian_noise_to_samples(erg.samples_B, None, erg.stddev_noise)
    label_A=np.zeros(number_of_samples)
    label_B=np.ones(number_of_samples)
    erg.noise_samples=np.concatenate((erg.noise_samples_A, erg.noise_samples_B))
    erg.noise_labels=np.concatenate((label_A,label_B))
    return erg


def generate_two_class_problem_with_unimportant(number_of_samples, dimensions_with_meaning, dimensions_meaningless, mean_range_A, mean_range_B, stddev_range_A, stddev_range_B, stddev_noise_range, seed=None):
    erg=generate_two_class_problem(number_of_samples,  dimensions_with_meaning, mean_range_A, mean_range_B, stddev_range_A, stddev_range_B, stddev_noise_range, seed)
    
    erg.means_C, erg.stddev_C = generate_feature_matrix_uniform(dimensions_meaningless, mean_range_B, stddev_range_B)
    samples_C1=draw_samples_from_distribution(erg.means_C, erg.stddev_C, number_of_samples)
    samples_C2=draw_samples_from_distribution(erg.means_C, erg.stddev_C, number_of_samples)
    
    erg.samples_A=np.concatenate((erg.samples_A,samples_C1),axis=1)
    erg.samples_B=np.concatenate((erg.samples_B,samples_C2),axis=1)
    erg.samples=np.concatenate((erg.samples_A, erg.samples_B))
    
    
    means_noise, erg.stddev_noise=generate_feature_matrix_uniform(dimensions_with_meaning+dimensions_meaningless, mean_range_A, stddev_noise_range)    
    erg.noise_samples_A=add_gaussian_noise_to_samples(erg.samples_A, None, erg.stddev_noise)
    erg.noise_samples_B=add_gaussian_noise_to_samples(erg.samples_B, None, erg.stddev_noise)
    
    erg.noise_samples=np.concatenate((erg.noise_samples_A, erg.noise_samples_B))
    return erg
    
def generate_validation_set(number_of_samples, erg):
    samples_A=draw_samples_from_distribution(erg.means_A, erg.stddev_A, number_of_samples)
    samples_B=draw_samples_from_distribution(erg.means_B, erg.stddev_B, number_of_samples)
    samples_C1=draw_samples_from_distribution(erg.means_C, erg.stddev_C, number_of_samples)
    samples_C2=draw_samples_from_distribution(erg.means_C, erg.stddev_C, number_of_samples)
    
    samples_A=np.concatenate((samples_A,samples_C1),axis=1)
    samples_B=np.concatenate((samples_B,samples_C2),axis=1)
    samples=np.concatenate((samples_A, samples_B))
    
    erg.validation_x=add_gaussian_noise_to_samples(samples, None, erg.stddev_noise)
    
    label_A=np.zeros(number_of_samples)
    label_B=np.ones(number_of_samples)
    erg.validation_y=np.concatenate((label_A,label_B))
    
    return erg
    
    