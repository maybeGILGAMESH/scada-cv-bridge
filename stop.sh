#!/bin/bash
# Остановить стенд: дашборд, окно просмотра, vmhdmi, ВМ.
DIR="$(cd "$(dirname "$0")" && pwd)"
pkill -f "[v]m/cvdash.py" && echo "дашборд остановлен"
pkill -f "[v]m/vmview.py" && echo "окно просмотра закрыто"
pkill -f "[v]mhdmi.py" && echo "vmhdmi остановлен"
"$DIR/vm/run-vm.sh" stop
