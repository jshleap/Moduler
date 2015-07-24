# Moduler
In this project you will find the tools to do shape modularity with a graph theory approach. It can also be extended to any correlation-based network.

Graph based Modularity.
This script will evaluate the data for modules. Such modules are defined as correlating variables, so the clustering 
is performed in the correlation space. It has an optional statistical significance test for the clustering and power
analysis of the result, as well as a bootstrap analysis. See options for more details.


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

E-mail: jshleap@dal.ca


In this version the the R dependencies have been extracted, and with them the RV coefficient test. In the near future it will be available in PyPy, so all the dependencies are sorted out.

Requirements:
=============

1. numpy module
2. scipy module
3. statsmodels module
4. matplotlib
5. scikit-learn   
6. PDBnet module: This is an open source module and can be found in :download: `LabBlouinTools<https://github.com/LabBlouin/LabBlouinTools>`
 
 *To install python modules 1-5 in UBUNTU: sudo apt-get install python-<module>	OR  sudo easy_install -U <module>  
 OR sudo pip install -U <module>*
 
 *For PDBnet, the whole directory should be downloded into a labblouin folder which should be in your pythonpath*
