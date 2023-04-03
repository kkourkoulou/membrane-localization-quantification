
Quantification of protein localization on vesicle membranes
-----------------------------------------------------------------------------------


This Python script calculates the radial and angular intensity profiles of GUVs 
encapsulating septin and/or actin and quantifies the corresponding protein
localization on the vesicle membrane. 


Localization is calculated as the difference between the median signal within the 
detected membrane width the average signal at a central area of the vesicle,
normalized by the median signal within the detected membrane width.
Negative values are clipped to 0.


Input
-----------------------------------------------------------------------------------

path_to_script  :    Specify the location of the script.
		     Note: Both main.py and skeleton.py should be in the same folder. 

Septin and Actin:    Boolean parameters 
		     If the corresponding protein is present, assign True, otherwise 
		     assign False. 	       

Analysis parameters: See related comments in the main.py script.

Function booleans:   Determine desired optional output apart from the localization
		     quantification. If optional output is desired, assign True, 
                     otherwise assign False. 
			  
Input files : Path and file names
-----------------------------------------------------------------------------------

The data sets to be analyzed should be stored in a subfolder named "data" contained 
in the folder of the script's path (path_to_data = path_to_script + "\\data").

For every image the files necessary to run the script are:
- tiff file of the membrane channel
- tiff file(s) of the protein(s) present
- csv file with DisGUVery's vesicle detection results

> In the csv DisGUVery file the "#" should be deleted from the first column's header ("# id_vesicle" should be renamed " id_vesicle").

The csv file name should be of the format:
"experiment_date-experiment_project_name-image_name-detected_vesicles"

> example: 20221012-sept_hex_pip2_0%-Region 1-detected_vesicles

The tiff file names should be of the format: 
"experiment_date-experiment_project_name-image_name-'...channel...'", where 
'...channel...' should be:
- 'C1' for the membrane.
- 'C2' for the single protein present or for the septin channel in case both proteins are
   present.
- 'C3' for the actin channel in case both proteins are present.

> example: 20221012-sept_hex_pip2_0%-Region 1-C1
 

Output
-----------------------------------------------------------------------------------
A file containing the experiment information, the vesicle coordinates, the backgrond 
signal estimated, the quantified localization and a comment detected artifacts is 
created in the path: "path_to_script + \\output" in a folder specific 
for each image analyzed. 
Additionally the intensity radial and angular profiles, the used masks and and image 
overview with annotated vesicle centres are saved as a png file, if requested by the user.

