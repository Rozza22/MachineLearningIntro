Data = readtable('EncodedData.csv'); % data already encoded on excel as was faster than on matlab
%DataArr = table2array(Data);

%% task2 - Data cleaning
RemovedColumnData = Data(:,2:8); % removed the first column as did not need the product code aswell as the type
imputedData = rmmissing(RemovedColumnData); %imputed the data

%% task2 - Feature engineering
DataMat = imputedData{:,:}; %converted the data from table to matrix form as MATLAB requires this for certain functions
StandardisedData = zscore(DataMat); %Standardised the data as this is essential for later parts of the pipeline

RPM = StandardisedData(:,4); %Named the columns what they are to help me be less confused
Torque = StandardisedData(:,5);

MOI = RPM./Torque; %Moment of inertia = Torque/speed of rotation

DataMat = [DataMat(:,1:5) MOI DataMat(:,6:7)]; %Put MOI into the dataset 

%% task2 - Pre-Processing - Standardising the data

[P_summary, Features, Classes, data_size] = data_statistics(imputedData);

copy_Data = Data(:,{'ProductType','AirTemperature_K_','ProcessTemperature_K_','RotationalSpeed_rpm_','Torque_Nm_','ToolWear_min_', 'FailureEncoded'});
statarray = grpstats(copy_Data,[],{'mean','min','max','std'}); %these two lines of code along with  line above allowed me to view all features alongside eachother
figure

% WeightedData
% Technique used here is to upsample the minority class. 
% I appended all the cases where 'L' was the product type to the end of my
% dataset.
% Minority class being Product type 'L' (encoded to 2)
ProductType = DataMat(:,1);
WeightedData = DataMat;
for j = 1:length(WeightedData(:,1))
    if ProductType(j) == 2
        WeightedData = [WeightedData; WeightedData(j,:)];
    end
end

%% task 3 - Regression model RPM->Torque

x = RPM; %rpm
y = Torque; %torque
N = length(x);

plot(x,y,'.')

xlabel('x');
ylabel('y');

%Create the dataset and set the percentage of data you want to use for
%validation
full_data_set = [x,y];
test_data_percentage = 0.2; %percentage of data used for testing
[train_data, test_data] = split_train_test(full_data_set,test_data_percentage);  



for m = 1:length(train_data)
    %Let us train a linear regression model first
    degree_of_polynomial = 3;  %Change this variable to decide what type of model to create. 
    % The value of 1 is a linear regression model and 2 and above polynomial model
    model_coefficients = polyfit(train_data(1:m,1), train_data(1:m,2),degree_of_polynomial);   %Change this line to 
    
    %Let us use the model coefficients to predict new values 
    %using the training dataset
    y_prediction_training = polyval(model_coefficients,train_data(1:m,1));
    
    %Let us use the model coefficients to predict new values 
    %using the test/validation dataset
    y_prediction_test = polyval(model_coefficients,test_data(:,1)); 
    
    training_errors = (y_prediction_training - train_data(1:m,2));
    training_rmse(m) = sqrt(mean((training_errors).^2)); %append results to end of training rmse array
    
    %Let us calculate both the validation and training errors and rmse
    test_errors = (y_prediction_test - test_data(:,2));
    test_rmse(m) = sqrt(mean((test_errors).^2)); %append results to end of validation rmse array
end

hold on
plot(test_data(:,1), y_prediction_test, 'o')
ylabel('Torque/Nm')
xlabel('RotationalSpeed')
legend('data set','predictions')

% figure
% plot(RPM,Torque,'.')
% hold on
% plot(RPMvsTorque1(:,1),RPMvsTorque1(:,2),'o')

figure
plot(1:length(training_rmse), training_rmse, 'r')
hold on
plot(1:length(test_rmse), test_rmse, 'b')
title('Original learning curve')
xlabel('Training set size');
ylabel('RMSE');
legend('training rmse','test rmse')

%% task 4/5 - cross validation - leave one out
%Use this section to set up a k-folds or leave out validation strategy

[train_indices,test_indices] = crossvalind('LeaveMOut',N,1); %Returns only indices which you now have to use
train_dataCV = full_data_set(train_indices, 1:end);
test_dataCV = full_data_set(test_indices, 1:end);

for i = 1:length(train_dataCV)
    %Let us train a linear regression model first
    degree_of_polynomialCV = 3;  %Change this variable to decide what type of model to create. 
    % The value of 1 is a linear regression model and 2 and above polynomial model
    model_coefficientsCV = polyfit(train_dataCV(1:i,1), train_dataCV(1:i,2),degree_of_polynomialCV);   %Change this line to 
    
    %Let us use the model coefficients to predict new values 
    %using the training dataset
    y_prediction_trainingCV = polyval(model_coefficientsCV,train_dataCV(1:i,1));
    
    %Let us use the model coefficients to predict new values 
    %using the test/validation dataset
    y_prediction_testCV = polyval(model_coefficientsCV,test_dataCV(:,1)); 
    
    training_errorsCV = (y_prediction_trainingCV - train_dataCV(1:i,2));
    training_rmseCV(i) = sqrt(mean((training_errorsCV).^2)); %append results to end of training rmse array
    
    %Let us calculate both the validation and training errors and rmse
    test_errorsCV = (y_prediction_testCV - test_dataCV(:,2));
    test_rmseCV(i) = sqrt(mean((test_errorsCV).^2)); %append results to end of validation rmse array
end

figure
plot(1:length(training_rmseCV), training_rmseCV, 'r')
hold on
plot(1:length(test_rmseCV), test_rmseCV, 'b')
title('Learning curve from LOOCV')
xlabel('Training set size');
ylabel('RMSE');
legend('training rmse','test rmse')

%% task 6 - Logistic regression to predict when machine will break
% NewWeightedData = [WeightedData(:,1:3) WeightedData(:,6:8)];
% 
% ProductType = WeightedData(:,1);
% NormProductType = ProductType - min(ProductType(:));
% %NormProductType = 
% 
% AirTemp = WeightedData(:,2);
% 
% ProcessTemp=WeightedData(:,3);
% 
% MOI = WeightedData(:,4);
% 
% ToolWear = WeightedData(:,5);
% 
% FailureEncoded = WeightedData(:,6);
% 
% Correlations = corrcoef(StandardisedData);
% 
% %for z = 1:length(NewWeightedData(:,1))
% WeightedPrediction = 0.1311*StandardisedData(:,1)-0.6807*NormalisedData(:,2)-0.6706*NormalisedData(:,3)-0.0663*NormalisedData(:,4)-0.2738*NormalisedData(:,5);
% %end

