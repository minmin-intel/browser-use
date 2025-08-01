1. Follow the webarena-setup repo readme to download docker images.
```bash
export WORKDIR=<your-work-directory>
git clone https://github.com/minmin-intel/webarena-setup.git
cd webarena-setup
git checkout test-webarena
```
2. Launch shopping admin
```bash
cd $WORKDIR/webarena-setup/webarena/
bash 02_docker_remove_containers.sh
bash 03_docker_create_containers.sh
bash 04_docker_start_containers.sh
bash 05_docker_patch_containers.sh # do not forget this step!!
```