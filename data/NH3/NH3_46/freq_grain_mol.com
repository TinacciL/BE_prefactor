%nosave
%nproc=24
%mem=24gb
%chk=input
#P ONIOM(B97D3/aug-cc-pvtz:external="xtb_clavigne") freq(noraman) 

 Title

0 1 0 1 0 1
H   -1       -1.779014     0.014382     2.493650 M
O   -1       -0.801503     0.048053     2.573257 M
H   -1       -0.521928    -0.834967     2.911681 M
H   -1       -0.396430    -1.842362    -2.816864 M
O   -1       -0.471278    -1.844696    -1.849844 M
H   -1       -1.419353    -1.752538    -1.658472 M
H   -1       -2.748063     1.885024     0.843330 M
O   -1       -2.649059     1.232541     0.129694 M
H   -1       -1.864497     1.511573    -0.376513 M
H   -1       -3.742801     1.924186    -3.028407 M
O   -1       -3.229991     2.259644    -3.800492 M
H   -1       -2.326707     2.400711    -3.498249 M
H   -1       -0.105470    -0.579747    -0.773861 M
O   -1        0.016364     0.161389    -0.149356 M
H   -1       -0.409430    -0.114033     0.672987 M
H   -1        0.954968    -1.930756    -4.743614 M
O   -1        0.017798    -2.113150    -4.563284 M
H   -1       -0.069131    -3.078210    -4.584791 M
H   -1       -4.947324     2.420617    -1.292755 M
O   -1       -4.578130     1.589622    -1.650216 M
H   -1       -3.900982     1.325794    -0.982831 M
H   -1       -3.559934     3.453815     2.462989 M
O   -1       -2.835624     2.833292     2.252980 M
H   -1       -2.064431     3.188467     2.732235 M
H   -1        2.554419     0.285767    -2.926639 M
O   -1        1.951289    -0.341631    -2.478175 M
H   -1        1.047271    -0.039223    -2.689512 M
H   -1       -1.139913     0.321822    -3.583703 M
O   -1       -0.335849     0.809192    -3.308990 M
H   -1       -0.586055     1.417540    -2.594229 M
H   -1        2.675904    -0.503696     0.595552 M
O   -1        2.583401    -1.261174    -0.018252 M
H   -1        2.320286    -0.894156    -0.883743 M
H   -1       -2.848825    -0.428360    -0.565259 M
O   -1       -2.984094    -0.906505    -1.394222 M
H   -1       -3.787459    -1.451238    -1.309236 M
H   -1        2.817668     1.309379    -0.803494 M
O   -1        1.996341     1.821430    -0.695822 M
H   -1        1.283569     1.174109    -0.510659 M
H   -1        0.559483     3.192248    -3.269606 M
O   -1        1.478434     3.015173    -3.029485 M
H   -1        1.482948     2.568452    -2.163386 M
H   -1        0.134155    -3.939031    -0.466323 M
O   -1       -0.102361    -4.636741    -1.119135 M
H   -1       -0.880845    -5.084832    -0.721332 M
H   -1       -6.249948     2.229614     3.000951 M
O   -1       -6.308411     2.447806     3.967928 M
H   -1       -7.057691     1.924414     4.314373 M
H   -1        1.151367    -3.639253     1.380898 M
O   -1        0.733726    -3.003875     0.770712 M
H   -1        1.412046    -2.348439     0.507403 M
H   -1       -2.908784     5.259246    -1.063420 M
O   -1       -2.726466     4.471769    -0.516936 M
H   -1       -3.613597     4.106211    -0.318149 M
H   -1       -5.505301    -1.195724     1.724364 M
O   -1       -5.233971    -0.571789     1.017757 M
H   -1       -5.340945    -1.045938     0.187784 M
H   -1       -0.332878    -3.092643     3.110416 M
O   -1       -0.026355    -2.274510     3.535590 M
H   -1        0.917972    -2.175530     3.329310 M
H   -1        5.247459     1.077603    -1.161775 M
O   -1        4.462173     0.551051    -0.954657 M
H   -1        4.693252     0.152396    -0.085714 M
H   -1        0.727957     2.473842     1.030874 M
O   -1        0.720501     2.152475     1.941733 M
H   -1        0.111840     1.394085     2.005352 M
H   -1        4.451973    -0.472408    -2.299309 M
O   -1        4.652315    -0.973318    -3.122183 M
H   -1        5.134478    -0.338874    -3.692321 M
H   -1        3.044813     3.306001    -1.031620 M
O   -1        3.922417     3.709999    -1.016692 M
H   -1        4.407959     3.404104    -1.806744 M
H   -1       -5.873746    -1.483937    -1.856526 M
O   -1       -5.343789    -2.175512    -1.403333 M
H   -1       -5.256475    -2.903007    -2.043247 M
H   -1       -4.906215    -3.992975     1.328469 M
O   -1       -5.595501    -3.369309     1.033832 M
H   -1       -5.365470    -3.099962     0.134344 M
H   -1        2.711354     2.148657    -3.679868 M
O   -1        3.386039     1.465686    -3.872458 M
H   -1        3.457496     1.410221    -4.830995 M
H   -1       -5.162810     3.966562     3.690740 M
O   -1       -4.592210     4.545815     3.174694 M
H   -1       -4.066638     5.091222     3.798390 M
H   -1        0.641400     0.982678    -4.726147 M
O   -1        1.132020     0.852468    -5.563508 M
H   -1        1.739094     0.102920    -5.390517 M
H   -1        1.024668     5.808162    -2.365354 M
O   -1        1.817795     5.491026    -1.900522 M
H   -1        2.372481     5.032317    -2.562296 M
H   -1       -2.781382    -3.458406    -1.712457 M
O   -1       -2.926228    -3.700722    -0.781537 M
H   -1       -3.694569    -4.287317    -0.743417 M
H   -1        2.078099     5.149104    -4.604940 M
O   -1        1.202755     5.229411    -5.022225 M
H   -1        1.192233     4.520374    -5.693044 M
H   -1        7.769020     1.894306    -0.918409 M
O   -1        6.981124     1.524712    -1.357539 M
H   -1        7.341800     0.829499    -1.944885 M
H   -1        4.502261     2.848225     0.314821 M
O   -1        4.836108     2.505696     1.168770 M
H   -1        4.763271     3.266012     1.782362 M
H   -1        1.071762    -3.155036    -2.111664 M
O   -1        1.930169    -3.007987    -2.533945 M
H   -1        2.016415    -2.033019    -2.582686 M
H   -1        1.181082     4.653086    -0.652964 M
O   -1        0.723436     4.194405     0.087731 M
H   -1        0.044737     4.808786     0.420014 M
H   -1       -5.743961     1.058617     1.241815 M
O   -1       -6.231406     1.888356     1.417302 M
H   -1       -7.163538     1.645926     1.184132 M
H   -1       -2.268441    -2.588850     0.318264 M
O   -1       -1.793318    -2.007142     0.941913 M
H   -1       -0.883026    -2.346968     0.944083 M
H   -1       -3.676687    -0.242355     4.292197 M
O   -1       -3.668964    -0.359688     5.269618 M
H   -1       -4.421997    -0.935045     5.534929 M
H   -1       -6.056997    -2.783763     3.281196 M
O   -1       -5.613985    -1.922237     3.354266 M
H   -1       -4.660870    -2.146899     3.356561 M
H   -1       -2.857205    -0.405671    -2.977216 M
O   -1       -2.595535    -0.486339    -3.921418 M
H   -1       -2.435018    -1.454212    -3.982623 M
H   -1       -3.435135    -3.412661    -3.653863 M
O   -1       -2.576954    -3.026303    -3.412538 M
H   -1       -1.882210    -3.662953    -3.674088 M
H   -1        3.225980    -1.173698    -4.234854 M
O   -1        2.673065    -1.305391    -5.028777 M
H   -1        3.305822    -1.486210    -5.742663 M
H   -1        4.674366     2.296044    -6.639477 M
O   -1        4.145297     1.472352    -6.639856 M
H   -1        3.307546     1.698628    -7.052168 M
H   -1       -4.183314     1.082982     5.783377 M
O   -1       -4.667433     1.911443     6.032053 M
H   -1       -5.232082     2.121221     5.260767 M
H   -1       -3.947999    -0.029517     2.018078 M
O   -1       -3.410207     0.296103     2.759927 M
H   -1       -3.328746     1.266723     2.663307 M
H   -1       -5.798201     0.577358    -2.237931 M
O   -1       -6.339153    -0.096466    -2.699482 M
H   -1       -5.911967    -0.203873    -3.571406 M
H   -1        1.902336    -3.828085    -4.065019 M
O   -1        1.837994    -4.295223    -4.920254 M
H   -1        1.461556    -5.166649    -4.706666 M
H   -1        4.484891    -3.516130    -0.589854 M
O   -1        4.761719    -2.755750    -0.031131 M
H   -1        3.997638    -2.133328    -0.043843 M
H   -1       -0.248004     3.056894    -0.924659 M
O   -1       -0.996355     2.671716    -1.410886 M
H   -1       -1.711759     3.330248    -1.290540 M
H   -1        1.759069     4.468839     1.516382 M
O   -1        2.213143     4.616703     2.365986 M
H   -1        2.024796     3.834613     2.908036 M
H   -1        3.705332     1.255062     1.639385 M
O   -1        2.935851     0.677069     1.808001 M
H   -1        2.162194     1.261391     1.927131 M
H   -1        3.156042    -3.935302    -2.018926 M
O   -1        3.856960    -4.553685    -1.699136 M
H   -1        4.534249    -4.573255    -2.392910 M
H   -1        4.982253    -3.784777    -4.468572 M
O   -1        5.413343    -3.456506    -3.641542 M
H   -1        5.004371    -2.598394    -3.410657 M
H   -1       -1.082564     5.129798     2.066490 M
O   -1       -1.263131     5.673348     1.264151 M
H   -1       -1.897172     5.167668     0.693260 M
H   -1       -5.690519     3.220401     0.456270 M
O   -1       -5.240642     3.714040    -0.255624 M
H   -1       -5.693449     4.567001    -0.419408 M
H   -1        5.032177    -3.763831     2.975949 M
O   -1        4.277861    -3.588811     2.407053 M
H   -1        4.609088    -3.315461     1.521692 M
H   -1        5.847042    -1.214801     1.363922 M
O   -1        5.446372    -0.323144     1.344262 M
H   -1        6.229650     0.259628     1.382775 M
H    0        1.650221    -6.656931    -3.058672 H
O    0        1.422475    -6.839647    -3.999484 H
H    0        2.295998    -7.005720    -4.398732 H
H   -1       -7.194587    -3.577785     0.843838 M
O   -1       -8.162310    -3.578055     0.643186 M
H   -1       -8.622073    -3.375848     1.487172 M
H   -1        3.242990    -2.338477     2.838232 M
O   -1        2.555667    -1.669136     3.018488 M
H   -1        2.776338    -0.869586     2.507365 M
H   -1       -1.916476     2.788100     5.080797 M
O   -1       -2.635332     3.431692     5.221811 M
H   -1       -3.374287     2.930762     5.615976 M
H   -1       -0.809603     0.997661     3.995016 M
O   -1       -0.689645     1.559703     4.793086 M
H   -1        0.148563     2.013553     4.653649 M
H   -1        0.257470     3.409770    -6.954458 M
O   -1        1.212504     3.215618    -6.831020 M
H   -1        1.231267     2.370528    -6.342742 M
H   -1        7.161871    -2.729848     4.741987 M
O   -1        6.407818    -3.187904     4.300257 M
H   -1        5.601290    -2.830777     4.741699 M
H   -1        7.995390     1.570133     1.380493 M
O   -1        7.913222     0.645680     1.681431 M
H   -1        8.400108     0.083135     1.050399 M
H   -1       -4.085749    -0.509277    -4.638497 M
O   -1       -5.014806    -0.556525    -4.972355 M
H   -1       -5.003956    -0.192617    -5.878863 M
H   -1        6.447846    -1.375560    -2.155787 M
O   -1        6.860392    -1.819266    -1.409847 M
H   -1        6.131259    -2.093420    -0.821993 M
H   -1        4.713207    -0.713581     4.447899 M
O   -1        5.122118     0.026300     3.968501 M
H   -1        5.070077    -0.150991     3.005029 M
H   -1       -0.020223     3.392956     2.856829 M
O   -1       -0.650340     4.050304     3.214591 M
H   -1       -0.339832     4.321663     4.100065 M
H   -1        8.247035    -1.147114    -0.706210 M
O   -1        9.034537    -0.764960    -0.259155 M
H   -1        9.734640    -1.428400    -0.329610 M
H   -1        9.626406    -3.694364     2.181126 M
O   -1        9.328109    -3.516293     1.263055 M
H   -1        10.092014    -3.358580     0.697535 M
H   -1       -2.387170    -2.508187     3.893678 M
O   -1       -3.010063    -2.608418     3.163730 M
H   -1       -2.556210    -2.286072     2.344493 M
H   -1       -2.165907     0.748010    -6.271186 M
O   -1       -2.796875     1.493172    -6.319010 M
H   -1       -3.048662     1.687484    -5.392266 M
H   -1       -1.244846     3.825103    -5.538720 M
O   -1       -1.368541     3.778450    -6.515282 M
H   -1       -1.956344     3.013522    -6.648818 M
H   -1       -4.582083    -6.210999    -1.063719 M
O   -1       -5.056088    -5.461980    -0.674308 M
H   -1       -5.826578    -5.294905    -1.259546 M
H    0        2.679839    -5.762981    -1.432158 H
O    0        1.854809    -6.289037    -1.346594 H
H    0        1.132275    -5.605239    -1.267807 H
H   -1        3.295515    -4.253790    -5.588558 M
O   -1        4.256022    -4.242394    -5.855964 M
H   -1        4.365201    -3.513773    -6.472675 M
H   -1        7.478026     3.567163     2.010203 M
O   -1        7.394932     3.199062     1.121447 M
H   -1        6.466086     2.863969     1.074753 M
H   -1        6.648605    -3.772498     0.378639 M
O   -1        7.139171    -4.584177     0.168399 M
H   -1        8.026880    -4.415206     0.540195 M
H   -1       -7.482527    -4.419831    -2.390774 M
O   -1       -7.249696    -5.345694    -2.155593 M
H   -1       -7.870481    -5.609795    -1.449697 M
H   -1       -4.394879     6.882276     4.857139 M
O   -1       -3.507667     6.546518     4.627858 M
H   -1       -3.023184     6.352010     5.444678 M
H   -1        7.268169     3.692202     4.777633 M
O   -1        7.266279     3.280190     3.903771 M
H   -1        7.522914     2.349827     4.051201 M
H   -1       -5.791069     1.017401     6.943464 M
O   -1       -6.558701     0.476605     7.242055 M
H   -1       -6.280388    -0.446069     7.118737 M
H   -1       -0.454247    -4.833734    -2.839741 M
O   -1       -0.628771    -4.749877    -3.794408 M
H   -1       -1.217832    -5.468226    -4.058827 M
H   -1        6.721684     0.378850     4.325200 M
O   -1        7.690865     0.573610     4.278340 M
H   -1        7.903349     0.518323     3.319713 M
H   -1        4.803514    -6.626875    -3.788768 M
O   -1        4.121158    -6.788869    -4.473245 M
H   -1        4.268562    -6.121373    -5.150725 M
H   -1       -5.909150    -1.907683    -5.206873 M
O   -1       -6.522559    -2.675411    -5.307275 M
H   -1       -7.377338    -2.261019    -5.593657 M
H   -1       -7.656157     0.752855     6.027261 M
O   -1       -8.219948     1.072978     5.278152 M
H   -1       -8.833318     1.703538     5.659184 M
H   -1       -5.242530     5.813628     2.230215 M
O   -1       -5.482797     6.721540     1.954085 M
H   -1       -4.627216     7.198436     1.928872 M
H   -1        5.770605     1.636573    -4.026116 M
O   -1        6.011330     0.847657    -4.558765 M
H   -1        5.516931     0.925722    -5.377071 M
H   -1       -5.886622     6.500127     0.305384 M
O   -1       -5.913377     6.261530    -0.644828 M
H   -1       -5.025035     6.497507    -0.986541 M
H    0       -0.248575    -7.542133    -3.278882 H
O    0       -0.957049    -7.717724    -2.633367 H
H    0       -0.498165    -7.819376    -1.789000 H
H   -1       -2.512691    -1.478561     5.552934 M
O   -1       -1.857299    -2.201078     5.702656 M
H   -1       -1.025277    -1.728196     5.893913 M
H   -1        7.538861    -4.425736    -1.495149 M
O   -1        7.660033    -4.420161    -2.468475 M
H   -1        6.879574    -3.958396    -2.820363 M
H   -1       -8.817179    -2.943488    -3.226578 M
O   -1       -7.895487    -2.855688    -2.891318 M
H   -1       -7.352783    -2.673035    -3.673564 M
H   -1        8.837759    -3.246303    -2.869439 M
O   -1        9.467716    -2.522006    -3.073730 M
H   -1        8.963287    -1.693271    -2.988812 M
H   -1       -5.072280    -4.965112    -3.506261 M
O   -1       -5.040676    -4.000744    -3.388785 M
H   -1       -5.562077    -3.618694    -4.105680 M
H   -1       -3.671761     4.013077    -3.569047 M
O   -1       -3.384726     4.872897    -3.220939 M
H   -1       -3.674811     5.561700    -3.820218 M
H   -1       -0.269324     4.750660    -4.102675 M
O   -1       -0.923487     4.059653    -3.895286 M
H   -1       -1.707693     4.507389    -3.518713 M
H   -1       -1.921057    -3.963024     5.422846 M
O   -1       -2.211636    -4.829149     5.096377 M
H   -1       -3.145968    -4.690983     4.861372 M
H   -1       -2.442486     8.385786    -3.852104 M
O   -1       -2.643253     7.431950    -3.944275 M
H   -1       -2.445061     7.183355    -4.874061 M
H   -1       -3.007022     7.402292     3.149764 M
O   -1       -2.936761     7.584830     2.197729 M
H   -1       -2.336485     6.891971     1.854172 M
H   -1        4.810078     1.687900     4.165026 M
O   -1        4.650479     2.639938     3.985163 M
H   -1        5.522753     3.000844     3.737700 M
H   -1        7.896894    -1.591315     6.264906 M
O   -1        8.337788    -1.800452     5.418420 M
H   -1        8.290752    -0.960225     4.920940 M
H   -1        6.042250    -1.328219     7.778921 M
O   -1        6.744101    -0.807873     7.375600 M
H   -1        6.281897    -0.064037     6.928549 M
H   -1       -1.770310     5.612266    -6.531874 M
O   -1       -1.908501     6.554968    -6.352171 M
H   -1       -1.039593     6.982171    -6.456963 M
H   -1       -8.185837    -0.112148     3.976030 M
O   -1       -8.000747    -0.659601     3.199916 M
H   -1       -7.094635    -1.001127     3.304271 M
H   -1       -8.518570     0.160629     0.219183 M
O   -1       -8.550008     0.810888     0.940363 M
H   -1       -8.519189     0.278731     1.753640 M
H   -1        0.846722    -0.404105     5.920307 M
O   -1        0.014347    -0.474036     6.444078 M
H   -1       -0.491632     0.293301     6.124823 M
H   -1       -0.495158     7.041999    -1.752573 M
O   -1       -0.453051     6.643255    -2.644887 M
H   -1       -1.309357     6.832690    -3.090374 M
H    0        0.646870    -8.054117     0.827916 H
O    0       -0.245908    -8.353565     0.544045 H
H    0       -0.731232    -8.380794     1.400274 H
H   -1        2.013433     2.805433     5.205533 M
O   -1        2.022134     2.510009     4.270473 M
H   -1        2.962554     2.532860     4.015362 M
H   -1       -7.968088    -5.557785    -5.410865 M
O   -1       -7.071925    -5.418210    -5.758519 M
H   -1       -6.922308    -4.464327    -5.721374 M
H    0        3.061673    -7.436594     1.102177 H
O    0        2.149909    -7.110553     1.113787 H
H    0        2.052517    -6.781580     0.171690 H
H   -1        3.288578     5.643012    -7.499678 M
O   -1        3.444546     4.680398    -7.606791 M
H   -1        2.617528     4.231019    -7.373356 M
H   -1       -0.543695     7.038336     0.417606 M
O   -1       -0.241591     7.735704    -0.188428 M
H   -1       -0.919322     8.425365    -0.145961 M
H   -1        4.430747     3.798422     5.166631 M
O   -1        4.343523     4.457481     5.897205 M
H   -1        5.225614     4.528549     6.271649 M
H   -1        7.644408     2.983070     6.757044 M
O   -1        7.113060     3.818658     6.669952 M
H   -1        7.599509     4.488505     7.149147 M
H    0        4.192398    -7.972204    -0.957937 H
O    0        4.661208    -7.434286    -0.277521 H
H    0        5.098528    -6.761492    -0.832419 H
H   -1        6.729897    -5.920316    -2.600378 M
O   -1        5.960385    -6.513193    -2.534765 M
H   -1        6.293500    -7.426123    -2.456420 M
H   -1       -8.986219    -2.053160     3.060758 M
O   -1       -9.467988    -2.877358     2.837187 M
H   -1       -10.207551    -2.583647     2.278369 M
H   -1       -2.709662     8.804160     0.885269 M
O   -1       -2.571258     9.147154    -0.013280 M
H   -1       -3.369900     9.703158    -0.190799 M
H   -1        8.104829     0.843915     7.335399 M
O   -1        8.442760     1.533374     6.753538 M
H   -1        8.293873     1.192241     5.854622 M
H   -1       -2.376350     5.275804     7.599054 M
O   -1       -2.098099     5.481201     6.707653 M
H   -1       -2.287918     4.665612     6.154401 M
H   -1       -6.288098    -6.422942    -3.201789 M
O   -1       -5.619617    -6.630901    -3.880078 M
H   -1       -6.061620    -6.359129    -4.713682 M
H   -1        6.796640    -2.675518     2.720176 M
O   -1        6.916859    -2.427436     1.784723 M
H   -1        7.863189    -2.507493     1.595320 M
H   -1        2.298988     6.751804    -0.934673 M
O   -1        2.552429     7.502196    -0.351531 M
H   -1        1.707947     7.841621    -0.030348 M
H   -1       -4.182866    -7.348516    -2.741950 M
O   -1       -3.604961    -7.388232    -1.970579 M
H   -1       -2.701678    -7.493913    -2.318715 M
H   -1        0.841195     6.673014    -5.730485 M
O   -1        0.664635     7.537572    -6.169456 M
H   -1        0.681600     8.182220    -5.432154 M
H   -1       -8.282232    -1.630270    -1.610623 M
O   -1       -8.573354    -1.348951    -0.729342 M
H   -1       -8.340337    -2.127465    -0.175163 M
H   -1       -3.304155     7.125489    -2.285300 M
O   -1       -3.399823     6.960812    -1.337525 M
H   -1       -3.098808     7.771965    -0.876719 M
H   -1        2.830997    -4.466052     2.397998 M
O   -1        1.917130    -4.813674     2.411311 M
H   -1        1.980804    -5.706992     2.013535 M
H   -1       -8.842327    -0.804430    -4.983375 M
O   -1       -8.753479    -1.394202    -5.768117 M
H   -1       -8.471354    -0.805253    -6.490209 M
H   -1        7.525267    -0.038221    -3.915369 M
O   -1        8.170855    -0.057872    -3.190105 M
H   -1        8.842144     0.605654    -3.402834 M
H   -1        4.191643     6.737219    -0.190321 M
O   -1        4.901849     6.101342    -0.373291 M
H   -1        4.444957     5.258887    -0.592070 M
H   -1       -7.970686     0.134720    -3.182263 M
O   -1       -8.868783     0.140977    -3.573253 M
H   -1       -9.469948    -0.082386    -2.861331 M
H   -1        6.096894     3.816296    -3.345599 M
O   -1        5.624040     3.035199    -3.003717 M
H   -1        6.214510     2.621812    -2.329747 M
H    0       -1.308015    -6.928725     0.243842 H
O    0       -1.912769    -6.177972     0.054838 H
H    0       -2.513726    -6.509442    -0.628370 H
H   -1       -3.946445    -7.736239     1.141587 M
O   -1       -4.166352    -7.469386     2.037045 M
H   -1       -3.956469    -6.505839     2.085991 M
H   -1       -2.854309    -5.025997     1.272465 M
O   -1       -3.565346    -4.902947     1.920711 M
H   -1       -3.285082    -4.173804     2.511647 M
H   -1        5.278452    -6.060593     4.910421 M
O   -1        4.521710    -5.828059     4.368997 M
H   -1        4.816408    -5.934633     3.448372 M
H   -1       -9.567753    -6.239627    -3.761645 M
O   -1       -9.584743    -5.794771    -4.639986 M
H   -1       -10.226046    -6.274655    -5.164466 M
H   -1        6.288582    -5.581704     1.297481 M
O   -1        5.778620    -6.162073     1.898968 M
H   -1        5.325152    -6.764518     1.294819 M
H   -1       -1.351336    -6.659539     2.699865 M
O   -1       -1.492208    -7.619764     2.766767 M
H   -1       -2.452569    -7.744471     2.762565 M
H   -1       -10.079669    -4.084943    -4.377725 M
O   -1       -10.209899    -3.155387    -4.125275 M
H   -1       -9.980732    -2.617163    -4.895136 M
H   -1        0.426672     7.949888    -3.375812 M
O   -1        0.743802     8.778670    -3.791707 M
H   -1        1.645202     8.927458    -3.437550 M
H   -1       -9.095586    -7.967225    -2.349527 M
O   -1       -9.685452    -7.212068    -2.363489 M
H   -1       -9.566608    -6.763037    -1.507037 M
H   -1        2.080365     0.780353     4.571743 M
O   -1        2.177913    -0.062743     5.054098 M
H   -1        2.391603    -0.753232     4.391230 M
H   -1       -6.574608     0.294870    -7.572655 M
O   -1       -7.533552     0.380745    -7.444294 M
H   -1       -7.689961     1.287337    -7.177098 M
H   -1        6.206467     3.828279    -7.348836 M
O   -1        5.618147     3.777065    -6.595522 M
H   -1        4.770427     4.228660    -6.895420 M
H   -1       -5.373827     10.330063     0.497213 M
O   -1       -4.883832     10.337489    -0.335903 M
H   -1       -5.469216     9.842602    -0.942511 M
H   -1       -0.816604    -1.172345    -5.721195 M
O   -1       -0.984478    -0.528739    -6.443722 M
H   -1       -0.200510     0.052355    -6.403210 M
H   -1        4.728902    -0.502684    -6.929419 M
O   -1        4.813767    -1.432197    -6.691811 M
H   -1        5.660817    -1.523900    -6.220540 M
H    0        2.654000    -9.317880    -2.166291 H
O    0        3.588133    -8.995729    -2.320302 H
H    0        3.547850    -8.418212    -3.095334 H
H   -1       -4.254604    -0.538670    -7.848329 M
O   -1       -4.765596     0.219186    -7.525442 M
H   -1       -4.087273     0.867463    -7.227362 M
H   -1        1.334612     3.901248     6.620449 M
O   -1        2.082437     3.311927     6.839588 M
H   -1        2.894390     3.835297     6.706993 M
H   -1       -11.339952    -2.622179     0.092151 M
O   -1       -10.922479    -1.977932     0.664157 M
H   -1       -10.192648    -1.592180     0.148205 M
H   -1       -7.417358     2.165363     7.593358 M
O   -1       -7.893232     3.002607     7.495452 M
H   -1       -8.011147     3.355936     8.376670 M
H   -1        6.213645    -9.005112    -0.997723 M
O   -1        6.295556    -9.147288    -1.945520 M
H   -1        5.399559    -9.362890    -2.246164 M
H   -1        6.584056     5.281481    -0.874321 M
O   -1        7.224686     4.614140    -1.141146 M
H   -1        7.417853     4.096223    -0.333102 M
H   -1       -5.356174    -2.710159     5.918667 M
O   -1       -5.371807    -1.936170     6.523605 M
H   -1       -4.742734    -2.174494     7.226521 M
H   -1        2.610275     1.672260     7.460362 M
O   -1        3.079584     0.828876     7.388985 M
H   -1        2.759505     0.426675     6.554159 M
H   -1       -2.692481    -2.678153     7.125281 M
O   -1       -3.178737    -2.797902     7.969529 M
H   -1       -2.557723    -2.455789     8.629011 M
H   -1        5.775288     2.222065     6.639508 M
O   -1        5.489556     1.342700     6.385447 M
H   -1        4.605526     1.196967     6.782740 M
H   -1        1.995645    -4.226333     4.218502 M
O   -1        2.477916    -4.147618     5.054116 M
H   -1        3.179838    -4.818584     4.981566 M
H   -1        10.428623    -2.679654    -1.744140 M
O   -1        10.926996    -2.669548    -0.881715 M
H   -1        11.855572    -2.646581    -1.106275 M
H   -1        3.848234     8.285195    -3.016425 M
O   -1        3.167788     8.864986    -2.652467 M
H   -1        2.995892     8.530673    -1.753210 M
H   -1       -8.149024    -6.430002     0.533258 M
O   -1       -8.796716    -6.020598    -0.062903 M
H   -1       -8.745533    -5.066063     0.189315 M
H   -1       -0.769641     9.659822    -3.412086 M
O   -1       -1.708000     9.826900    -3.201698 M
H   -1       -1.772853     9.963609    -2.254036 M
H   -1        4.098676     4.001913    -3.818210 M
O   -1        3.544578     4.787396    -3.855713 M
H   -1        4.104700     5.567183    -3.660463 M
H   -1        5.145122     6.616243    -2.064798 M
O   -1        5.073226     6.808151    -3.020133 M
H   -1        5.855130     6.444415    -3.455222 M
H   -1       -0.399075     5.348028     6.020078 M
O   -1        0.321079     4.863708     5.592145 M
H   -1        1.010941     5.502687     5.290534 M
H   -1        3.183415     5.720078     5.069199 M
O   -1        2.418285     6.151592     4.660567 M
H   -1        2.365711     5.803971     3.756146 M
H   -1       -7.212662     9.007082    -0.298192 M
O   -1       -6.923257     8.767081    -1.188296 M
H   -1       -6.688478     7.823534    -1.142630 M
H   -1       -1.290734    -4.917010     3.702389 M
O   -1       -0.846966    -4.850333     2.825973 M
H   -1        0.075749    -5.099147     2.928334 M
H   -1        4.244052    -1.963633     6.300418 M
O   -1        4.242033    -2.135094     5.334394 M
H   -1        3.530879    -2.788576     5.174965 M
H   -1        7.128867    -6.034491     3.303192 M
O   -1        7.505818    -5.653880     4.106809 M
H   -1        7.029957    -4.813192     4.242555 M
H   -1        9.046763    -4.680678     4.016229 M
O   -1        9.677764    -3.951583     3.866695 M
H   -1        9.409633    -3.240157     4.461063 M
H   -1        3.016085     7.840747    -8.174936 M
O   -1        3.022392     7.318603    -7.374040 M
H   -1        2.150341     7.475311    -6.936800 M
H   -1       -0.582322    -1.111692     8.164032 M
O   -1       -0.787581    -1.735231     8.874779 M
H   -1       -0.074782    -2.380171     8.860864 M
H   -1        8.343048     4.516833    -2.333749 M
O   -1        9.018959     4.530045    -3.059657 M
H   -1        9.743183     5.052354    -2.714674 M
H   -1        6.356627     4.658457    -5.238532 M
O   -1        6.679187     5.071937    -4.416015 M
H   -1        7.626778     4.917822    -4.328221 M
H   -1        10.631442     2.159068    -4.208548 M
O   -1        9.897462     2.060459    -3.604098 M
H   -1        9.514265     2.953920    -3.456655 M
H   -1        3.163135    -2.213444     7.959638 M
O   -1        3.994183    -1.699979     7.951268 M
H   -1        3.714646    -0.773731     8.011109 M
H   -1       -2.434510    -2.097201    -8.705537 M
O   -1       -2.843183    -1.752564    -7.912497 M
H   -1       -2.138799    -1.285632    -7.412765 M
H   -1        5.083464     5.488032     1.457925 M
O   -1        4.880396     4.946559     2.231138 M
H   -1        3.911238     4.971522     2.331836 M
H   -1       -3.747814    -6.360943    -4.663206 M
O   -1       -2.941099    -5.848810    -4.752605 M
H   -1       -3.171173    -5.083318    -5.302959 M
H   -1       -6.581693     6.291659     4.855968 M
O   -1       -6.148842     7.111672     4.554615 M
H   -1       -6.018554     7.010010     3.585663 M
H   -1       -7.466226     4.281867     6.115241 M
O   -1       -7.209342     4.696400     5.282016 M
H   -1       -6.962640     3.958651     4.701047 M
H   -1       -7.292176     9.569041     2.324649 M
O   -1       -7.052546     9.944155     1.452545 M
H   -1       -7.523445     10.775696     1.387353 M
H   -1       -5.940229    -6.880299     1.891289 M
O   -1       -6.556460    -6.261292     1.470239 M
H   -1       -6.022606    -5.830160     0.769164 M
H   -1       -3.744039    -4.516449     7.766638 M
O   -1       -3.976385    -5.369549     7.357625 M
H   -1       -3.185121    -5.648811     6.882640 M
H   -1        9.541583     0.895442    -0.565268 M
O   -1        9.587463     1.847722    -0.768451 M
H   -1        9.893346     1.898422    -1.680109 M
H   -1       -7.942533    -4.113674     3.403086 M
O   -1       -7.015971    -4.375272     3.408379 M
H   -1       -6.888726    -5.029589     2.692111 M
H   -1        1.080581    -2.111061     6.980076 M
O   -1        1.582480    -2.843178     7.354429 M
H   -1        1.796692    -3.428508     6.611080 M
H   -1       -3.450028    -2.981199    -6.723764 M
O   -1       -3.860759    -3.597000    -6.098918 M
H   -1       -4.809775    -3.468979    -6.181488 M
H   -1       -8.719301     8.474052     3.760144 M
O   -1       -7.930107     9.010426     3.849127 M
H   -1       -7.249320     8.425567     4.243723 M
H   -1        8.007479    -2.140462    -5.122308 M
O   -1        7.082734    -1.889178    -5.146401 M
H   -1        6.618201    -2.522973    -4.566918 M
H   -1       -4.862440    -4.762988     5.933561 M
O   -1       -4.911616    -4.196871     5.138406 M
H   -1       -5.676912    -4.458003     4.606280 M
N    0        1.105108    -10.124564    -1.731351 H
H    0        0.393179    -10.003592    -2.447011 H
H    0        0.681501    -9.821783    -0.854329 H
H    0        1.285222    -11.120882    -1.651326 H

