v4l2-ctl --device=/dev/video0 --set-ctrl white_balance_temperature_auto=0	#Desactiva ajuste auto de tempetarua 0 off 1 On
v4l2-ctl --device=/dev/video0 --set-ctrl white_balance_temperature=3500		#min=2000 max=6500
v4l2-ctl --device=/dev/video0 --set-ctrl exposure_auto=1			#Desactiva tiempo de exposicion automatico 0 off 1 ON
v4l2-ctl --device=/dev/video0 --set-ctrl exposure_absolute=10		#Min=3 max=2047
v4l2-ctl --device=/dev/video0 --set-ctrl gain=100				#min=0 max=255
v4l2-ctl --device=/dev/video0 --set-ctrl focus_auto=0				#Ajuste automatico de enfoque 0 off 1 On
v4l2-ctl --device=/dev/video0 --set-ctrl focus_absolute=38			#min=0 max=250 step=5
v4l2-ctl --device=/dev/video0 --set-ctrl brightness=130				#min=0 max=255 step=1
