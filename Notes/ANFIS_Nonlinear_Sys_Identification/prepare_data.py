# -*- coding: utf-8 -*-
"""
Copyright (C) Tue Nov 29 18:25:39 2016  Jianshan Zhou
Contact: zhoujianshan@buaa.edu.cn	jianshanzhou@foxmail.com
Website: <https://github.com/JianshanZhou>

This program is free software: you can redistribute
 it and/or modify it under the terms of
 the GNU General Public License as published
 by the Free Software Foundation,
 either version 3 of the License,
 or (at your option) any later version.
 
This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY;
 without even the implied warranty of MERCHANTABILITY
 or FITNESS FOR A PARTICULAR PURPOSE.
 See the GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with this program.
 If not, see <http://www.gnu.org/licenses/>.
 
This module processes the original data.
"""
import numpy as np
import matplotlib.pyplot as plt

def load_test_data1():
    data = np.loadtxt("sys_data.csv")
    sample_num = 500
    training_data = []
    all_data = []
    for i in range(sample_num):
        x1 = data[100+i]
        x2 = data[106+i]
        x3 = data[112+i]
        x4 = data[118+i]
        x = np.array([x1,x2,x3,x4])
        y = np.array([data[124+i]])
        training_data.append((x,y))
        all_data.append((x,y))
        
    validation_data = []
    for i in range(sample_num):
        x1 = data[600+i]
        x2 = data[606+i]
        x3 = data[612+i]
        x4 = data[618+i]
        x = np.array([x1,x2,x3,x4])
        y = np.array([data[624+i]])
        validation_data.append((x,y))
        all_data.append((x,y))

    return training_data,validation_data,all_data


def _test1():
    data = np.loadtxt("sys_data.csv")
    print data.shape
    labelfont = {"family":"serif","size":20}
    plt.figure(0,figsize=(9,8))
    plt.plot(data,'-r',lw=8.0, label="Fuel comsuption")
    plt.grid(True)
    plt.legend(prop={"size":labelfont["size"],"family":"serif"})  
    xlabelstr = "Epoch $k$"
    ylabelstr = "Fuel level"
    plt.xlabel(xlabelstr,fontdict=labelfont)
    plt.ylabel(ylabelstr,fontdict=labelfont)
    plt.xticks(fontsize=labelfont["size"],fontname=labelfont["family"])
    plt.yticks(fontsize=labelfont["size"],
               family=labelfont["family"])
    plt.show()

if __name__ == "__main__":
    load_test_data1()
