%Function I use to print the images.
%The user just passes the names of the files as arguments and the images are printed.
%The files passed as arguments are the ones we get from C.
% It can also return the data of the images in Arrays, if the user wants to.
% Recommended syntax:
%[image, noised_image, denoised_image,residuals_image]=print_results("image.txt", "noised_image.txt", "denoised_image.txt");


function [image, noised_image, denoised_image,residuals_image]= print_results(image_name, noised_name, denoised_name)

        %First getting the image data into arrays
        
        %The image should be loaded from the C file,which gives us the data in one column. This is why 
        %I have to use these three following commands:
        image=load(image_name);          
        im_dim=sqrt(length(image));
        image=reshape(image', im_dim, im_dim);
        image=image';           
        
        noised_image=load(noised_name);
        noised_image=reshape(noised_image', im_dim, im_dim);
        noised_image=noised_image';
        
        denoised_image=load(denoised_name);
        denoised_image=reshape(denoised_image', im_dim, im_dim);
        denoised_image=denoised_image';
        
        residuals_image=denoised_image-noised_image;
        %Finished getting the image data into arrays

        %Then, using the arrays to print the images
        a=figure('Name', 'Original Image')
        imagesc(image); 
        axis image;
        colormap gray;
        
        b=figure('Name', 'Noised Image')
        imagesc(noised_image);
        axis image;
        colormap gray;
        
        c=figure('Name','Denoised Image');
        imagesc(denoised_image);
        axis image;
        colormap gray;
        
        d=figure('Name','The Residuals');
        imagesc(residuals_image);
        axis image;
        colormap gray;
        %Finished printing
        
        
        saveas(a, "original.jpg")
        saveas(b, "noised.jpg")
        saveas(c, "denoised_image.jpg")
        saveas(d, "residuals_image.jpg")
        

end