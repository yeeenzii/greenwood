"""
Database Helper
_________
This module has a class for database to handle operations
"""
#        :param connection_string: str
import oracledb
from env_vars import env_dict

class Database(object):

    """Database Class which is responsible for managing connections and queries to the database"""

    def __init__(self):
        """
        Initalizes the database connection
        """

        self.connection = None
        if self.__connect():  #TODO manage failure
            print("Connected to database")
        else:
            print("Could not connect to database")
        #Initialize database schema if database it does not exist
        #connection.

    def __connect(self) -> True:
        """
        This method is used to connect to database
        :return: bool
        """
        connection = oracledb.connect(
            user=env_dict['DB_USER'],
            password=env_dict['DB_PASSWORD'],
            dsn=env_dict['DB_CONNECTION_STRING'])
        self.connection = connection;
        return True

    def query(self, query):
        with self.connection.cursor() as cursor:
            cursor.execute(query)
        pass
    def create_organization(self):
        pass
    def invite_user(self):
        pass
    def create_user(self):
        pass
    def remove_user(self):
        pass
    def add_activity_record(self):
        pass

# # Create a table
# with connection.cursor() as cursor:
#
#     cursor.execute("""
#         begin
#             execute immediate 'drop table todoitem';
#             exception when others then if sqlcode <> -942 then raise; end if;
#         end;""")
#
#     cursor.execute("""
#         create table todoitem (
#             id number generated always as identity,
#             description varchar2(4000),
#             creation_ts timestamp with time zone default current_timestamp,
#             done number(1,0),
#             primary key (id))""")
#
#     print("Table created")
#
# # Insert some data
# with connection.cursor() as cursor:
#
#     rows = [ ("Task 1", 0 ),
#              ("Task 2", 0 ),
#              ("Task 3", 1 ),
#              ("Task 4", 0 ),
#              ("Task 5", 1 ) ]
#
#     cursor.executemany("insert into todoitem (description, done) values(:1, :2)", rows)
#     print(cursor.rowcount, "Rows Inserted")
#
# connection.commit()
#
# # Now query the rows back
# with connection.cursor() as cursor:
#
#     for row in cursor.execute('select description, done from todoitem'):
#         if (row[1]):
#             print(row[0], "is done")
#         else:
#             print(row[0], "is NOT done")

database = Database();
database.query("SHOW DATABASES;")