import sqlite3


class MeasurementSaver:
    def __init__(self, database_path="/Users/gghosal/Desktop/gaurav_new_photos/measurements.db"):
        self.connection = sqlite3.connect(database_path)
        self.cursor = self.connection.cursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.commit()

        self.connection.close()

    def save_singular_measurement(self, date: int, time: int, potnumber: str, position: str, area: float) -> None:
        self.cursor.execute('''INSERT INTO plants VALUES (?,?,?,?,?)''',
                            tuple((int(date), int(time), str(potnumber), str(position), float(area))))
        self.connection.commit()

    def save_multiple_measurements(self, array):
        """Array should be an array of multiple [[date,time,potnumber, position,area]]"""
        self.cursor.execute('''INSERT INTO plants VALUES (?,?,?,?,?)''',
                            array)
        self.connection.commit()

    def create_database(self, path):
        # conn = sqlite3.connect(path)
        # cursor = conn.cursor()
        self.cursor.execute('''CREATE TABLE plants 
                            (date integer, time integer, potnumber text, position text, area real)''')
        self.connection.commit()
