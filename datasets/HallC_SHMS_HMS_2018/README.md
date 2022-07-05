
# Hall-C SHMS and HMS Spectrometer Dataset for early 2018

The file _HallC_SpectrometerAngles2018.dat_ was generated
on gluon47 with the command

~~~ bash
myData -b "2018-01-15" -e "2018-02-28" -m history ecHMS_Angle ecSHMS_Angle > HallC_SpectrometerAngles2018.dat
~~~

The angles are in degrees.

<hr>

Here is a plot from _MyaViewer_ of the values vs. time. The viewer was started (on gluon47) with:

~~~
MyaViewer -mhistory
~~~

![MYA plot of angles](MYAview.png?raw=true "MYAview plot of HMS and SHMS angles for early 2018")

