function showDigits(rawData)
% param rawData : input data must be 256*1 data
% usage: 'showDigits(digits_data(1,:));'
% show the digits by picture
    data=reshape(rawData,16,16);

    % subplot 1*2 grid
    subplot(1,2,1);
    % show the data in black-white
    imshow(data);
    title('black-white');
    
    subplot(1,2,2);
    % show the data in gray 
    imshow(data,[0,255]);
    title('gray');