# import oracledb
# from env_vars import env_dict
#
# def database_helper():
#     connection = oracledb.connect(
#         user="ADMIN",
#         password="1Vr%125lBlhf",
#         dsn="(description= (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=adb.ap-singapore-1.oraclecloud.com))(connect_data=(service_name=g6c4c0c6d559479_greenwood_low.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))")  # the connection string copied from the cloud console
#
#     print("Successfully connected to Oracle Database")
#
#
# if __name__ == "__main__":
#     database_helper()
#
# #
# # #"0Fm*56sWpSTO"
# # #l04C5a%m%DWO
# # #adb.ap-singapore-1.oraclecloud.com/g6c4c0c6d559479_greenwood_high.adb.oraclecloud.com
# #
# # import getpass
# #
# # # pw = getpass.getpass("Enter password: ")
# #
# #
# # # Create a table
# # with connection.cursor() as cursor:
# #
# #     cursor.execute("""
# #         begin
# #             execute immediate 'drop table todoitem';
# #             exception when others then if sqlcode <> -942 then raise; end if;
# #         end;""")
# #
# #     cursor.execute("""
# #         create table todoitem (
# #             id number generated always as identity,
# #             description varchar2(4000),
# #             creation_ts timestamp with time zone default current_timestamp,
# #             done number(1,0),
# #             primary key (id))""")
# #
# #     print("Table created")
# #
# # # Insert some data
# # with connection.cursor() as cursor:
# #
# #     rows = [ ("Task 1", 0 ),
# #              ("Task 2", 0 ),
# #              ("Task 3", 1 ),
# #              ("Task 4", 0 ),
# #              ("Task 5", 1 ) ]
# #
# #     cursor.executemany("insert into todoitem (description, done) values(:1, :2)", rows)
# #     print(cursor.rowcount, "Rows Inserted")
# #
# # connection.commit()
# #
# # # Now query the rows back
# # with connection.cursor() as cursor:
# #
# #     for row in cursor.execute('select description, done from todoitem'):
# #         if (row[1]):
# #             print(row[0], "is done")
# #         else:
# #             print(row[0], "is NOT done")


"""
Greenwood
_________
Main launcher for greenwood forest management system
This is a module which has a class for database to handle operations
"""


class Database(object):

    """Database Class which is responsible for connecting to database"""

    def __init__(self, connection_string):
        """
        This is a connection string to connect to oracle sql server
        :param connection_string: str
        """
        self.connection_string = connection_string

    def connect(self) -> True:
        """
        This method is used to connect to database
        :return: bool
        """
        print("connected to database")
        return True