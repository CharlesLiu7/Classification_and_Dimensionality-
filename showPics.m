function showPics(rawData,varargin)
% param rawData : input data must be 2500*1 data
% param save_name : save the picture as the name of save_name
% usage: 'showDigits(train_data(1,:));'
% show the faces by picture
    data=reshape(rawData,50,50);

    % subplot 1*2 grid
    subplot(1,2,1);
    % show the data in black-white
    imshow(data);
    title('face flexible');
    
    subplot(1,2,2);
    % show the data in gray 
    imshow(data,[0,1]);
    title('face 0-1');
    
    if(length(varargin)~=0)
        imwrite(data,varargin{1});
    end
    