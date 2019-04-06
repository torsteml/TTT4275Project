function formant_table = load_vowdata(filename)
%% Import data from text file.
% Script for importing data from the following text file:
%
%    TTT4275\- Project\- Vowels\Wovels\Wovels\vowdata_nohead.dat

%% Initialize variables.
if nargin < 1
    filename = 'vowdata_nohead.dat';
end

%% Format for each line of text:
%   column1: text (%s)
%	column2: double (%f)
%   column3: double (%f)
%	column4: double (%f)
%   column5: double (%f)
%	column6: double (%f)
%   column7: double (%f)
%	column8: double (%f)
%   column9: double (%f)
%	column10: double (%f)
%   column11: double (%f)
%	column12: double (%f)
%   column13: double (%f)
%	column14: double (%f)
%   column15: double (%f)
%	column16: double (%f)
%   column17: categorical (%C)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%5s%4f%4f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%C%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string',  'ReturnOnError', false);

%% Remove white space around all cell columns.
dataArray{1} = strtrim(dataArray{1});

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.



%% Create output variable
formant_table = table;
formant_table.file = cellstr(dataArray{:, 1});

vowel_code = cell(height(formant_table),1);
talker_group_code = cell(height(formant_table),1);
talker_number = zeros(height(formant_table),1);
for i = 1:height(formant_table)
    filename = formant_table.file{i};
    vowel_code{i} = filename(4:end);
    talker_group_code{i} = filename(1);
    talker_number(i) = str2double(filename(2:3));
end
formant_table.talker = categorical(talker_group_code);
formant_table.talker_number = talker_number;
formant_table.vowel = categorical(vowel_code);
formant_table.vowel_duration = dataArray{:, 2};
formant_table.F0s = dataArray{:, 3};
formant_table.F1s = dataArray{:, 4};
formant_table.F2s = dataArray{:, 5};
formant_table.F3s = dataArray{:, 6};
formant_table.F4s = dataArray{:, 7};
formant_table.F120 = dataArray{:, 8};
formant_table.F220 = dataArray{:, 9};
formant_table.F320 = dataArray{:, 10};
formant_table.F150 = dataArray{:, 11};
formant_table.F250 = dataArray{:, 12};
formant_table.F350 = dataArray{:, 13};
formant_table.F180 = dataArray{:, 14};
formant_table.F280 = dataArray{:, 15};
formant_table.F380 = dataArray{:, 16};

end