x = [10
100
1000
10000];

y = [10
100
1000
10000
100000];

z1 = [3.5E-05	7.7E-05	0.001357	0.00143	0.00321	0.00247	0.003452	0.003748	0.004917
0.000325	0.000341	0.001654	0.002829	0.003063	0.002933	0.004141	0.004165	0.004266
0.003295	0.002733	0.003888	0.005006	0.006689	0.008712	0.005757	0.005418	0.009123
0.033151	0.0419	0.026887	0.024693	0.044097	0.026944	0.030069	0.029098	0.050804
0.544761	0.49072	0.435894	0.404797	0.405385	0.454384	0.450567	0.384483	0.264935
];


for i = 1:9
    semilogx(y,z1(:,1)./z1(:,i), 'linewidth', 1.2);
    hold on;
end

legend("p1","p2","p3","p4","p5","p6","p7","p8","p9", 'Location','northwest');
title("N = 100");