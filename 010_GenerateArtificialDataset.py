# -*- coding: utf-8 -*-
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
import numpy as np
import pandas as pd
import noisyFeaturesArtificalDataset as nfad

no_of_samples=150
meaningfull_variables=20
overall_variables=4000
oversampling_factor=6

data=nfad.generate_two_class_problem_with_unimportant(no_of_samples,
                                                      meaningfull_variables, 
                                                      overall_variables-meaningfull_variables, 
                                                      [-.5,.5],[-.5,.5],[0.3,2],[0.3,2],[0.3,2],
                                                      seed=21)
data=nfad.generate_validation_set(1000,data)

train_data=np.concatenate((data.samples_A, data.samples_B))
noise_data=nfad.draw_samples_from_distribution(np.zeros(train_data.shape[1]),data.stddev_noise,1)
train_data=train_data+noise_data
data.noise_samples=train_data

data.validation_y=data.validation_y.reshape([-1,1])
data.noise_labels=data.noise_labels.reshape([-1,1])

df_noisefree_training=pd.DataFrame(nfad.add_gaussian_noise_to_samples(data.samples, None, data.stddev_noise))
df_noisefree_training['Y']=pd.Series(data.noise_labels.reshape([-1]))
df_noisefree_training['Groups']=pd.Series(range(data.noise_labels.shape[0]))

df_gwn_training=pd.DataFrame(data.noise_samples)
df_gwn_training['Y']=pd.Series(data.noise_labels.reshape([-1])*2-1)
df_gwn_training['Groups']=pd.Series(range(data.noise_labels.shape[0]))

df_gwn_validation=pd.DataFrame(data.validation_x)
df_gwn_validation['Y']=pd.Series(data.validation_y.reshape([-1])*2-1)
df_gwn_validation['Groups']=pd.Series(range(data.noise_labels.shape[0]))
