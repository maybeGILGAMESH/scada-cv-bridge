#!/bin/bash
# Подготовка ВМ: диск поверх образа Debian + seed.iso для cloud-init.
# ./setup-vm.sh          — создать (если ещё нет)
# ./setup-vm.sh --reset  — пересоздать ВМ с нуля
set -euo pipefail
cd "$(dirname "$0")"
BASE=debian-13-generic-amd64.qcow2
for b in qemu-system-x86_64 qemu-img cloud-localds; do
  command -v $b >/dev/null || { echo "нет $b: sudo apt install -y qemu-system-x86 qemu-utils cloud-image-utils"; exit 1; }
done
if [ ! -f "$BASE" ]; then
  echo "скачиваю образ Debian 13 (~420 МБ)…"
  URL=https://cloud.debian.org/images/cloud/trixie/latest
  curl -fL -o "$BASE.part" "$URL/$BASE"
  SUM=$(curl -fsSL "$URL/SHA512SUMS" | awk -v f="$BASE" '$2 == f {print $1}')
  echo "$SUM  $BASE.part" | sha512sum -c - && mv "$BASE.part" "$BASE"
fi
[ "${1:-}" = --reset ] && rm -f scada-vm.qcow2 seed.iso
[ -f ssh_key ] || ssh-keygen -q -t ed25519 -N '' -C scada-vm -f ssh_key
if [ ! -f scada-vm.qcow2 ]; then
  qemu-img create -q -f qcow2 -F qcow2 -b "$BASE" scada-vm.qcow2 20G
  sed "s|__SSH_KEY__|$(cat ssh_key.pub)|" user-data.tmpl > user-data
  cloud-localds seed.iso user-data meta-data
  echo "ВМ создана. Первый запуск ставит пакеты (~3–5 мин) и перезагружается сам."
fi
