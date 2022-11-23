# Greenwood
Forest management system to alarm authorities of logging, poaching and track wild elephant movement.

## How to Contribute (Backend)
Create mariadb_database image and start container
```
docker run --name greenwood_mariadb -e MYSQL_ROOT_PASSWORD=pass -p 3308:3308 -d docker.io/library/mariadb:10.10
```
Connect to greenwood_mariadb container via TTY
```
docker exec -it greenwood_mariadb bash
```

## Literature 
//abstract// </br>
Read the full report [here](https://safaa.dev/greenwood_paper)

## License
<p align="center">
                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed. [Read More...](https://github.com/Safayy/greenwood/blob/main/LICENSE)
</p> 

## Contibutors and Acknowledgement
Created by: Safa Yousif </br>
Team Members: Michelle ... </br>
Acknowledgements to Dr.Marina Ng, our supervisor and our clients; Dr.Ee Phin Wong, Dr.Yen Yi Loo, Mr.Noah Thong. Special thanks to Mr. Naufal Rahman for his support.
