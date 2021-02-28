%Matlab function used to convert ONE image to a csv file (and give us the .csv file).
% You can convert both an image from a file (for example .jpg), or you can also convert an array to a .csv
% The .csv that are created depict the image in one column (every row has only on element), and that is 
% because the C function reading the .csv is implemented to work with such csv files.

%Inputs: 1)image_name-> The name of the image file
%            2)file_name  -> The name of the file (the .csv) to be created.
%            3)flag         -> Used to decide on whether to convert a file or an array
%    IF flag==1 convert array
%    ELSE convert a file
%           4)array -> The name of the array to be converted (must be loaded on the Workspace)

% Typical syntax: get_csv("64.jpg", "file_name.csv", 0)           %To convert from a file named 64.jpg
%                           get_csv("", "file_name.csv", 1, array_name)     %To convert an array named array_name
% In the case of an image conversion, the image has to be black/ white (no RGB values).

function get_csv(image_name, file_name, flag, array)

    if(flag==1)         %Case of array 
        array_reduced=array';
        
        %check whether the image is not square
        array_dims=size(array);
        if(array_dims(1)~=array_dims(2))
            fprintf("The picture is not a square array! Give me a square one!\n");
        end
        %Finished checking thether it is not squared
        
        array_reduced=array_reduced(:);
        csvwrite(file_name, array_reduced);
    else                        %Case of file
        
        my_image=imread(image_name);      %Reading the image. The user specifies the name of the file.
        a=cast(my_image,'double');      % Converting the integer values to doubles
        
        %Checking whether the image is not black and white and whether it is not square
        if(ndims(a)>2)
            fprintf("Pass a black/white image and then call the function!");
            return
        end
        image_dims=size(a);
        if(image_dims(1)~=image_dims(2))
            fprintf("The picture is not a square array! Give me a square one!\n");
        end
        %Finished checking.
        
        a=a./255;                                     % Normalising
        a_vector=reshape(a.',1,[]);         %Converting the whole array to one column (this is the format we use for the C implementation)
        a_vector=a_vector';
        csvwrite(file_name,a_vector);        %Writing the data to a csv file, with the name the user wants.
    end
        
end





