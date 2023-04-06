clc
clear 
close all

%% load files
% filename = 'sensors.csv';
filename = 'sensors_Tag4.csv';
sensor_data = readtable(filename,'PreserveVariableNames',true);

% crop data range
%sensor_data = sensor_data(1:1000,:);


%% post-process
%Time = sensor_data.("Timestamp(ms)");
%DateTime = datetime(2022,1,1,-4,0,sensor_data.("Timestamp(ms)"));
% offset 4 hours (14400 seconds) to convert GMT to Atlantic Standard Time (AST)
time = datetime(1e-3*sensor_data.("Timestamp(ms)")-14400, 'convertfrom', 'posixtime', 'Format', 'MM/dd/yy HH:mm:ss.SSS');

depth = -10*sensor_data.("Pressure bar");

[yaw, pitch, roll] = quat2angle([sensor_data.Quat_Re sensor_data.Quat_i sensor_data.Quat_j sensor_data.Quat_k]);

roll_deg = roll*57.2958;
pitch_deg = pitch*57.2958;
yaw_deg = yaw*57.2958;
%% plot

nPlots = 6;

subplot(nPlots,1,1)
plot(time, depth)
ylabel("Depth [m]")

subplot(nPlots,1,2)
plot(time, sensor_data.AmbientLight)
ylabel("Light [count]")

subplot(nPlots,1,3)
plot(time, sensor_data.("BoardTemp degC"),time, sensor_data.("WaterTemp degC"))
ylabel("Temperature [C]")
legend('Board', 'Water')

subplot(nPlots,1,4)
plot(time, sensor_data.("Batt_V1 V"),time, sensor_data.("Batt_V2 V"))
ylabel("Battery [V]")
legend('Batt 1', 'Batt 2')

subplot(nPlots,1,5)
plot(time, sensor_data.("Batt_I mA"))
ylabel("Current [mA]")

subplot(nPlots,1,6)
plot(time, roll_deg,time, pitch_deg,time, yaw_deg)
ylabel("IMU [degrees]") % ^{\circ}
legend('Roll','Pitch','Yaw')

% subplot(nPlots,1,6)
% plot(time, sensor_data.State)
% ylabel("State machine [-]")

