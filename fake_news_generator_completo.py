import pandas as pd
import random
import csv

def modificar_dataset(dataset_path, nombres_originales, nombres_nuevos, partidos_originales, partidos_nuevos, resultado_modificado, resultado_no_modificado):
    # Leer el dataset original
    df = pd.read_csv(dataset_path, sep=';', encoding='utf-8')
    
    # Crear un diccionario de mapeo entre nombres originales y nombres nuevos aleatorios
    nombres_mapping = dict(zip(nombres_originales, [random.choice(nombres_nuevos) for _ in range(len(nombres_originales))]))
    partidos_mapping = dict(zip(partidos_originales, [random.choice(partidos_nuevos) for _ in range(len(partidos_originales))]))
    
    # Aplicar la transformación en los campos 'Titulo' y 'Descripcion'
    df['Titulo'] = df['Titulo'].replace(nombres_mapping, regex=True)
    df['Descripcion'] = df['Descripcion'].replace(nombres_mapping, regex=True)
    df['Titulo'] = df['Titulo'].replace(partidos_mapping, regex=True)
    df['Descripcion'] = df['Descripcion'].replace(partidos_mapping, regex=True)
    
    # Filtrar las filas que tuvieron algún reemplazo
    df_reemplazado = df[df['Titulo'].str.contains('|'.join(nombres_mapping.keys())) |
                        df['Descripcion'].str.contains('|'.join(nombres_mapping.keys())) |
                        df['Titulo'].str.contains('|'.join(partidos_mapping.keys())) |
                        df['Descripcion'].str.contains('|'.join(partidos_mapping.keys()))]
    
    # Filtrar las filas que no tuvieron ningún reemplazo
    df_no_reemplazado = df[~df['Titulo'].str.contains('|'.join(nombres_mapping.keys())) &
                           ~df['Descripcion'].str.contains('|'.join(nombres_mapping.keys())) &
                           ~df['Titulo'].str.contains('|'.join(partidos_mapping.keys())) &
                           ~df['Descripcion'].str.contains('|'.join(partidos_mapping.keys()))]
    
    # Escribir los resultados en dos archivos CSV con campos Titulo y Descripcion
    df_reemplazado.to_csv(resultado_modificado, sep=';', index=False, encoding='utf-8')
    df_no_reemplazado.to_csv(resultado_no_modificado, sep=';', index=False, encoding='utf-8')

# Ejemplo de uso
dataset_path = './GenerateFakeNews/soloTrue.csv'
resultado_modificado = './GenerateFakeNews/soloTrue_modificado.csv'
resultado_no_modificado = './GenerateFakeNews/soloTrue_no_modificado.csv'

nombres_originales = ['Abascal', 'Abel Caballero', 'Ada Colau', 'Adam Przeworski', 'Adolfo Suárez', 'Adriana Lastra', 'Adrián Vázquez', 'Agustín Moreno', 'Aitor Martínez', 'Albert Boadella', 'Albert Comellas', 'Albert Garcia', 'Albert Noguera', 'Albert Rivera', 'Alberto Alcocer', 'Alberto Casero', 'Alberto Fernández', 'Alberto Garzón', 'Alberto Luceño', 'Alberto Núñez Feijóo', 'Alberto Reyero', 'Alberto Rodríguez', 'Alberto San Juan', 'Alejandra Jacinto', 'Alfonso Bauluz', 'Alfonso Grau', 'Alfonso Guerra', 'Alfonso Rojo', 'Alfonso Rueda', 'Alfonso Serrano', 'Alfredo Montoya', 'Alicia Sánchez-Camacho', 'Ana Pontón', 'Ana Rosa Quintana', 'Ana Torroja', 'Ana Varela', 'Ander Gil', 'Anna Erra', 'Anna Gabriel', 'Anne Hidalgo', 'Antonia Morillas', 'Antonio Cerdá', 'Antonio Fernández', 'Antonio Garamendi', 'Antonio Hernando', 'Antonio Miguel Carmona', 'Antonio Muñoz', 'Antonio Vergara', 'Antonio de la Torre', 'Aragonès', 'Arias Navarro', 'Arkaitz Rodríguez', 'Arnaldo Otegi', 'Arturo González Panero', 'Aurora Picornell', 'Ayuso', 'Aznar', 'Bakartxo Ruiz', 'Baltasar Garzón', 'Barbón', 'Beatriz Gimeno', 'Begoña Gómez', 'Belarra', 'Bernie Sanders', 'Blas Infante', 'Blas de Lezo', 'Bolaños', 'Boluarte', 'Boris Johnson', 'Borja San Ramón', 'Borja Sémper', 'Brahim Ghali', 'Bárcenas', 'Carles Puigdemont', 'Carlos Arenas', 'Carlos Bardem', 'Carlos Fabra', 'Carlos García Adanero', 'Carlos García Juliá', 'Carlos Lesmes', 'Carlos Martín', 'Carlos Martínez-Almeida', 'Carlos Mazón', 'Carlos Prieto', 'Carlos Rojas', 'Carlos Sánchez Mato', 'Carlota Sales', 'Carmen Calvo', 'Carmen Fúnez', 'Carmen Martínez Aguayo', 'Carolina Alonso', 'Carolina Darias', 'Carolina Elías', 'Carrero Blanco', 'Casado', 'Celia Cánovas', 'Celia Mayer', 'Celia Villalobos', 'Cipriano Martos', 'Clara Campoamor', 'Clara Ponsatí', 'Colón de Carvajal', 'Corinna Larsen', 'Cosidó', 'Cospedal', 'Cristina Cifuentes', 'Cristina Fallarás', 'Cristina Fernández', 'Cristina Fernández de Kirchner', 'Cristina Narbona', 'Cristina Seguí', 'Cristina Zalba', 'Cuca Gamarra', 'Cuixart', 'Cándido Conde-Pumpido', 'Daniel Guzmán', 'Daniel Ripa', 'Daniel Sirera', 'David Beriain', 'David Leo', 'David Lucas', 'David Moya', 'David Simon', 'David Suárez', 'David Urdín', 'David de la Cruz', 'Delcy Rodríguez', 'Diana Morant', 'Diana Riba', 'Diego Cañamero', 'Dina Bousselham', 'Dióscoro Galindo', 'Dolores Delgado', 'Dolors Sabater', 'Donald Trump', 'Díaz Ayuso', 'Edmundo Bal', 'Eduard Pujol', 'Eduardo Rubiño', 'Eduardo Zaplana', 'Elías Bendodo', 'Emilio Hellín', 'Enrique Arnaldo', 'Enrique Cayuela', 'Enrique López', 'Enrique Ossorio', 'Enrique Santiago', 'Enriqueta Chicano', 'Ernest Maragall', 'Ernest Urtasun', 'Ernesto Alba', 'Esperanza Aguirre', 'Esperanza Casteleiro', 'Esperanza Gómez', 'Esteban', 'Esteban Beltrán', 'Esteban González Pons', 'Esteban Álvarez', 'Esther López', 'Eugenio Merino', 'Eugenio Pino', 'Eva Granados', 'Eva Kaili', 'Fadel Breica', 'Fatima Hamed', 'Fausto Canales', 'Feijoo', 'Feijoó', 'Felipe Alcaraz', 'Felipe González', 'Felipe Sicilia', 'Felipe VI', 'Fermín Muguruza', 'Fernandez Díaz', 'Fernando Barcia', 'Fernando Plaza', 'Fernando Presencia', 'Fernando Valdés', 'Fernández Díaz', 'Fernández Vara', 'Florentino Pérez', 'Forcadell', 'Fran Ferri', 'Francesc Vallès', 'Francisco', 'Francisco Camps', 'Francisco Correa', 'Francisco Cuenca', 'Francisco Franco', 'Francisco Fuentes', 'Francisco González', 'Francisco Igea', 'Francisco Martín', 'Francisco Martínez', 'Franco', 'Fèlix Larrosa', 'Fèlix Millet', 'Félix Bolaños', 'Félix López Rey', 'Gabriel Boric', 'Gabriel Rufián', 'Gallardo', 'Galán', 'Gandía Arturo Torró', 'Gara Santana', 'Georg Michael Welzel', 'Gerardo Iglesias', 'Gerardo Pisarello', 'Gijón', 'Gisbert', 'Gloria Calero', 'Gloria Elizo', 'Gloria Martín', 'Gloria Steinem', 'Gonzalo Berger', 'Gonzalo Boye', 'Gonzalo Caballero', 'Gonzalo Capellán', 'Gonzalo Pérez Jácome', 'Gonzalo Santonja', 'González Laya', 'González Pons', 'Gregorio Ordóñez', 'Griñán', 'Guillermo Fernández Vara', 'Gustavo Petro', 'Hana Jalloul', 'Helena Maleno', 'Héctor Gómez', 'Héctor Illueca', 'Ian Gibson', 'Iglesias', 'Ignacio Aguado', 'Ignacio Cembrero', 'Ignacio González', 'Ignacio Ramonet', 'Ildefonso Falcones', 'Inma Nieto', 'Inmaculada Nieto', 'Inés Arrimadas', 'Inés Herreros', 'Ione Belarra', 'Irene Lozano', 'Irene Montero', 'Irune Costumero', 'Isa Serra', 'Isabel Bonig', 'Isabel Díaz Ayuso', 'Isabel II', 'Isabel Peralta', 'Isabel Rodríguez', 'Isidoro Moreno', 'Iván Duque', 'Jaume Asens', 'Jaume Collboni', 'Jaume Graells', 'Jaume Matas', 'Javier Arenas', 'Javier Ayala', 'Javier Biosca', 'Javier Imbroda', 'Javier Lambán', 'Javier Losada', 'Javier Madrazo', 'Javier Maroto', 'Javier Negre', 'Javier Ruiz', 'Jesús Aguirre', 'Jesús Domínguez', 'Jesús Gil', 'Jesús Jurado', 'Jiménez Losantos', 'Joan Baldoví', 'Joan Fuster', 'Joan Ribó', 'Joan Subirats', 'Joaquim Bosch', 'Joaquim Forn', 'Joaquín Leguina', 'Joaquín Prat', 'Joe Biden', 'Jon Iñarritu', 'Jordi Cuixart', 'Jordi Martí', 'Jordi Puigneró', 'Jordi Pujol', 'Jordi Solé', 'Jordi Sànchez', 'Jordi Évole', 'Jorge Azcón', 'Jorge Fernández Díaz', 'Jorge Ignacio Palma', 'Josep Borrell', 'Josep Bou', 'Josep Maria Estela', 'Josep Maria Montaner', 'Josep Piqué', 'Josep Pujol', 'Josep Rius', 'Josep Rull', 'Josep Sunyol', 'José Antonio Griñán', 'José Antonio Nieto', 'José Barrionuevo', 'José Bernal', 'José Bono', 'José García Buitrón', 'José Ignacio García', 'José Luis Moreno', 'José Luis Peñas', 'José Luis Ábalos', 'José Manuel Bandrés', 'José Manuel Franco', 'José Manuel Miñones', 'José Manuel Soria', 'José María Cataluña', 'José María González', 'José Miñones', 'José Monzo', 'José Primo de Rivera', 'José de Francisco', 'Juan Antonio Delgado', 'Juan Bernardo Fuentes', 'Juan Bravo', 'Juan Carlos', 'Juan Carlos Aparicio', 'Juan Carlos Campo', 'Juan Carlos García Goena', 'Juan Carlos Girauta', 'Juan Carlos I', 'Juan Carlos Monedero', 'Juan Enciso', 'Juan Espadas', 'Juan García-Gallardo', 'Juan Ignacio Campos', 'Juan Lobato', 'Juan Marín', 'Juan Miguel Villar Mir', 'Juan Pedro Yllanes', 'Juan Rodríguez Poo', 'Juan de la Cierva', 'Juana Rivas', 'Juana Ruiz', 'Juanjo Carmona', 'Juanma Guijo', 'Juanma Moreno', 'Juanma del Olmo', 'Julian Assange', 'Julio Anguita', 'Julián Grimau', 'Julián Martínez', 'Junqueras', 'Justa Freire', 'Lambán', 'Largo Caballero', 'Laura Caldito', 'Laura Díez', 'Laura Luelmo', 'Laura Martín', 'Leggeri', 'Lenin', 'Leonor Paqué', 'Leonor de Borbón', 'Leopoldo López', 'Leticia Sabater', 'Letizia', 'Letizia Ortiz', 'Lidia Rubio', 'Lilith Verstrynge', 'Lizz Truss', 'Lluc Salellas', 'Lluis Apesteguia', 'Lluís Apesteguia', 'Lluís Rabell', 'Lluïsa Moret', 'Lorena Roldán', 'Lourdes Cebollero', 'Lourdes Maldonado', 'Lucía Figar', 'Luis Bárcenas', 'Luis Cueto', 'Luis García Montero', 'Luis Medina', 'Luis Olivera', 'Luis Tudanca', 'Luis Ángel Hierro', 'López Aguilar', 'López Obrador', 'Macarena Olona', 'Madrid Beltrán Gutiérrez', 'Manolo Mata', 'Manu Castro', 'Manu Pérez', 'Manuel Castells', 'Manuel Clavero Arévalo', 'Manuel Fraga', 'Manuel Gerena', 'Manuel Lapeña', 'Manuel Marchena', 'Manuel Prado', 'Manuel Ruiz', 'Manuela Bergerot', 'Manuela Carmena', 'Mar García Puig', 'Marcel Moore', 'Marcial Dorado', 'Marcos Ana', 'Marga Ferré', 'Margarita Robles', 'Mariano Rajoy', 'Mariano Sánchez Soler', 'Mariano Veganzones', 'Maribel Mora', 'Mario Vaquerizo', 'Marta Calvo', 'Marta Higueras', 'Marta Rovira', 'Marta del Castillo', 'Martina Velarde', 'Juanma Moreno', 'Martín González', 'Martín Villa', 'Maru Díaz', 'María Asencio', 'María Barrero', 'María Botto', 'María González Veracruz', 'María Guardiola', 'María Marín', 'María Rozas', 'María Teresa Fernández de la Vega', 'María Teresa Pérez', 'Matilde Eiroa', 'Mauricio Casals', 'Maza', 'Melisa Rodríguez', 'Mercedes González', 'Mercedez González', 'Mercé Gironés', 'Meri Pita', 'Miguel Anxo Fernández Lores', 'Miguel Buch', 'Miguel Campos', 'Miguel Delibes', 'Miguel González', 'Miguel Hernández', 'Miguel Urbán', 'Miguel Ángel Blanco', 'Miguel Ángel Rodríguez', 'Mikel Antza', 'Mikel Torres', 'Mikel Zabalza', 'Millán Astray', 'Miquel Pueyo', 'Mireia Comas', 'Mireia Vehí', 'Mohamed VI', 'Monago', 'Monedero', 'Montero', 'Montesinos', 'Montás', 'Mónica García', 'Mónica González', 'Móstoles', 'Nacho Calle', 'Nadia Calviño', 'Naim Darrechi', 'Nancy Pelosi', 'Narcís Serra', 'Narváez', 'Ni Belarra', 'Nieto Castro', 'Néstor Rego', 'Núria Marín', 'Obama', 'Odón Elorza', 'Olatz Vázquez', 'Oriol Junqueras', 'Ortega Cano', 'Ortega Smith', 'Ossorio', 'Pablo Bustinduy', 'Pablo Casado', 'Pablo Crespo', 'Pablo Echenique', 'Pablo Fernández', 'Pablo González', 'Pablo Hasel', 'Pablo Ibar', 'Pablo Iglesias', 'Pablo Montesinos', 'Pablo Motos', 'Pablo Zapatero', 'Paco Bezerra', 'Paco Espinosa', 'Paco Vázquez', 'Pacto de Toledo', 'Paloma Alonso', 'Paloma Fernández Coleto', 'Pamela Palenciano', 'Paqui Maqueda', 'Pardo Bazán', 'Pasqual Maragall', 'Patricia Guasp', 'Patricio P. Escobal', 'Pau Fons', 'Pau Ricomà', 'Paz Esteban', 'Pedraz', 'Pedro Antonio Sánchez', 'Pedro Arriola', 'Pedro Campos', 'Pedro Castillo', 'Pedro Cortés', 'Pedro González-Trevijano', 'Pedro Mouriño', 'Pedro Quevedo', 'Pedro Sanz', 'Pedro Sánchez', 'Pedro Varela', 'Pedro del Cura', 'Pepe Vélez', 'Pepe Álvarez', 'Pepu Hernández', 'Pilar Alegría', 'Pilar Garrido', 'Pilar Lima', 'Pilar Llop', 'Plácido Domingo', 'Pío García Escudero', 'Quim Torra', 'Rafael Vera', 'Rajoy', 'Ramiro de Maeztu', 'Ramón Luis Valcárcel', 'Ramón Tamames', 'Ramón de Carranza', 'Raquel Sánchez', 'Raúl Solís', 'Raül Blanco', 'Raül Romeva', 'Repsol Suárez', 'Revilla', 'Rey Juan Carlos', 'Reyes Maroto', 'Ribó', 'Ricardo Melchior', 'Rita Barberá', 'Rita Maestre', 'Rivera', 'Roberto Fraile', 'Roberto Lakidain', 'Roberto Saviano', 'Roberto Sotomayor', 'Robles', 'Rocío Delgado', 'Rocío Monasterio', 'Rodrigo Cuevas', 'Rodrigo Rato', 'Rodrigo Torrijos', 'Rodríguez Fraga', 'Rodríguez Palop', 'Roger Torrent', 'Roldán', 'Rosa Díez', 'Rosa García Alcón', 'Rosa Pérez', 'Rosa Pérez Garijo', 'Rosalía Iglesias', 'Royuela', 'Ruiz de Gordoa', 'Rupérez', 'Ruth Porta', 'Rutte', 'Salvador Alba', 'Salvador Illa', 'Salvador Martín Valdivia', 'Samuel Luiz', 'Sánchez', 'San Chin Choon', 'San Juan Nepomuceno', 'Sandra Gómez', 'Sandra Heredia', 'Santiago Abascal', 'Santos Cerdán', 'Scott Morrison', 'Sebastian Kurz', 'Sergio Gómez Reyes', 'Sofía Castañón', 'Sol Sánchez', 'Soledad Castillero', 'Soledad Luque', 'Soledad Murillo', 'Sonia Vivas', 'Susana Díaz', 'Susana Hornillo', 'Suso Díaz', 'Tania Sánchez', 'Tania Varela', 'Teo García Egea', 'Teodoro García Egea', 'Teresa Bueyes', 'Teresa Jiménez Becerril', 'Teresa Ribera', 'Teresa Rodríguez', 'Teresa Rodríguez Rubio', 'Tito Berni', 'Tomás Díaz Ayuso', 'Toni Cantó', 'Toni Morillas', 'Toni Nadal', 'Toni Rodon', 'Toni Valero', 'Tornero', 'Torres', 'Trias', 'Unai Sordo', 'Unai Urruzuno', 'Utrera Molina', 'Vallejo', 'Vargas Llosa', 'Vicente Lertxundi', 'Vicente del Bosque', 'Vicenç Villatoro', 'Vicky Rosell', 'Victoria Landa', 'Victoria Rosell', 'Videla', 'Víctor Gutiérrez', 'Víznar', 'Waldino Varela', 'Willy Toledo', 'Xavier Domènech', 'Xavier García Albiol', 'Xavier Rius', 'Xavier Trias', 'Ximo Puig', 'Xiomara Castro', 'Xosé Sánchez Bugallo', 'Yassin Kanjaa', 'Yeremi Vargas', 'Yolanda Diaz', 'Zapatero', 'Zelenski', 'Francisco Vázquez', 'Paco Vázquez', 'Abel Caballero', 'Ada Colau', 'Adolfo Suárez', 'Adolfo Suárez Illana', 'Adriana Lastra', 'Adrián Barbón', 'Aitor Esteban', 'Albert Boadella', 'Albert Rivera', 'Alberto Garzón', 'Alberto Ruiz-Gallardón', 'Alfonso Alonso', 'Alfonso Guerra', 'Alfredo Pérez Rubalcaba', 'Alicia Sánchez-Camacho', 'Ana Botella', 'Ana Botín', 'Ana Mato', 'Ana Oramas', 'Ana Palacio', 'Ana Pastor', 'Ana Rosa Quintana', 'Ander Gil', 'Andrea Levy', 'Anna Gabriel', 'Antoni Comín', 'Antonio Baños', 'Antonio Hernando', 'Antonio Machado', 'Antonio Maíllo', 'Antonio Sanz', 'Arancha González Laya', 'Artur Mas', 'Baltasar Garzón', 'Beatriz Corredor', 'Begoña Gómez', 'Belarra', 'Bermúdez de Castro', 'Bono', 'Borja Sémper', 'Borrell', 'Boti García', 'Boya', 'Bousselham', 'Bétera', 'Calviño', 'Calvo', 'Camilo de Dios', 'Cani Fernández', 'Carla Antonelli', 'Carles Campuzano', 'Carles Mulet', 'Carles Puigdemont', 'Carme Forcadell', 'Carmela Silva', 'Carmen Alborch', 'Carmen Calvo', 'Carmen Castilla', 'Carmen Franco Polo', 'Carmen Martínez-Bordiú', 'Carmen Montón', 'Carmen Polo', 'Carmen Torres', 'Carmen de la Peña', 'Carolina Darias', 'Casado', 'Abascal', 'Cayetana Álvarez de Toledo', 'Celia Villalobos', 'Clara Campoamor', 'Cristina Cifuentes', 'Cs Ángel Garrido', 'Díaz Ayuso', 'Díez Picazo', 'Edmundo Bal', 'Fernando Clavijo', 'Fernando Martín', 'Fernando Martínez López', 'Fernando Miramontes', 'Fernando Roig', 'Fernando Román', 'Fernando Savater', 'Fernando Simón', 'Fernando Valdés', 'Fernández Díaz', 'Fernández Vara', 'Florentino Pérez', 'Francis Franco', 'Francisco Camps', 'Francisco Correa', 'Francisco Franco', 'Francisco González', 'Francisco Granados', 'Francisco Igea', 'Francisco Javier Guerrero', 'Francisco Javier Sánchez Gil', 'Francisco Martínez', 'Francisco Serrano', 'Francisco Vázquez', 'Gabriel Rufián', 'Gaspar Llamazares', 'Gerardo Iglesias', 'Gonzalo Caballero', 'González Laya', 'González Pons', 'Griñán', 'Guindos', 'Gómez', 'Gómez-Reino', 'Hana Jalloul', 'Ignacio Aguado', 'Ignacio Escolar', 'Ignacio Garriga', 'Ignacio González', 'Iglesias', 'Inés Arrimadas', 'Ione Belarra', 'Iratxe García', 'Irene Lozano', 'Irene Montero', 'Isa Serra', 'Isabel Díaz Ayuso', 'Iturgaiz', 'Iván Espinosa de los Monteros', 'Iñigo Errejón', 'Iñigo de la Serna', 'Javier Arenas', 'Javier Maroto', 'Javier Nart', 'Javier Negre', 'Javier Ortega Smith', 'Javier Solana', 'Javier Zarzalejos', 'Jenaro Castro', 'Jenn Díaz', 'Jenner López Escudero', 'Jesús Montero', 'Jesús Muñecas', 'Jesús Sepúlveda', 'Joan Baldoví', 'Joan Coscubiela', 'Joan Garcés', 'Joan Herrera', 'Joan Mena', 'Joan Mesquida', 'Joan Ribó', 'Joan Subirats', 'Joaquim Forn', 'Joaquín Leguina', 'Joaquín Pérez Rey', 'Jordi Alemany', 'Jordi Borràs', 'Jordi Cuixart', 'Jordi Montull', 'Jordi Pujol', 'Jordi Pujol Ferrusola', 'Jordi Salvador', 'Jordi Sevilla', 'Jordi Sànchez', 'Jordi Turull', 'Jordi Xuclà', 'Jorge Azcón', 'Jorge Fernández Díaz', 'Joseba Pagazaurtundúa', 'Josep Borrell', 'Josep Lluis Núñez', 'Josep Lluis Trapero', 'Josep Piqué', 'Josep Rull', 'Josu Erkoreka', 'José Antonio Sánchez', 'José Blanco', 'José Bono', 'José Couso', 'José Guirao', 'José Luis Ábalos', 'José María Aznar', 'José María García', 'José María Marco', 'José Ramón Bauzá', 'José Ángel Fernández Villa', 'Juan Carlos Campo', 'Juan Carlos Girauta', 'Juan Carlos I', 'Juan Carlos Monedero', 'Juan Carlos de Borbón', 'Juan Cotino', 'Juan Espadas', 'Juan Genovés', 'Juan Guaidó', 'Juan José Cortes', 'Juan José Tamayo', 'Juan Luis Cebrián', 'Juan Luis Rubenach', 'Juan María González', 'Juan Marín', 'Juan Muñoz', 'Juan Romero', 'Juan Rosell', 'Juan Trinidad', 'Juanma Moreno', 'Juanma Romero', 'Juanma Serrano', 'Julio Anguita', 'Julio Rodríguez', 'Junqueras', 'Jéssica Albiach', 'Lamela', 'Largo Caballero', 'Largo Mayo', 'Laura Borrás', 'Laura Duarte', 'Leopoldo López', 'Luis Bárcenas', 'Luis De Guindos', 'Manuel Azaña', 'Manuel Fraga', 'Manuela Carmena', 'Mariano Rajoy', 'Marta Pascal', 'Marta Rivera de la Cruz', 'Marta Rovira', 'Martín Villa', 'Martínez Almeida', 'María Teresa Fernández de la Vega', 'Melisa Rodríguez', 'Meritxel Batet', 'Miguel Ángel Blanco', 'Miguel Ángel Revilla', 'Miguel Ángel Rodríguez', 'Milagrosa Martínez', 'Miquel Buch', 'Miquel Iceta', 'Miquel Roca', 'Mónica García', 'Nadia Calviño', 'Narcís Serra', 'Núria Marín', 'Oriol Junqueras', 'Oriol Pujol', 'Ortega Smith', 'Otegi', 'Pablo Casado', 'Pablo Iglesias', 'Pablo Echenique', 'Pablo Hasel', 'Pablo Ibar', 'Paco Ferrándiz', 'Paco Frutos', 'Paco Guarido', 'Pedro Duque', 'Pedro García Aguado', 'Pedro J. Ramírez', 'Pedro Quevedo', 'Pedro Rollán', 'Pedro Santisteve', 'Pedro Sánchez', 'Pepe Mujica', 'Pepu Hernández', 'Pere Navarro', 'Pilar Rahola', 'Piqué', 'Primo de Rivera', 'Pujol', 'Pío García Escudero', 'Quim Forn', 'Quim Torra', 'Rajoy', 'Ramón Espinar', 'Ramón Jáuregui', 'Raquel Martínez', 'Raquel Romero', 'Rita Barberá', 'Rita Maestre', 'Roberto Fraile', 'Rocío Monasterio', 'Rodrigo Rato', 'Rosa Díez', 'Ruth Beitia', 'Salvador Illa', 'Santiago Abascal', 'Santiago Vidal', 'Susana Díaz', 'Soraya Rodríguez', 'Soraya Sáenz', 'Teresa Ribera', 'Teodoro García Egea', 'Uxue Barkos', 'Valerio', 'Víctor Barrio', 'Xabier Arzalluz', 'Xavi Hernández', 'Xavier Domènech', 'Xavier García Albiol', 'Ximo Puig', 'Xosé Manuel Beiras', 'Xulio Ferreiro', 'Yolanda Díaz', 'Yolanda González', 'Zapatero', 'Zerolo', 'Àngel Ros', 'Álvaro Lapuerta', 'Ángel Acebes', 'Ángel Gabilondo', 'Ángel Garrido', 'Ángel Viñas', 'Ángel Víctor Torres', 'Ángela Aguilera', 'Ángeles Pedraza', 'Íñigo Errejón']

nombres_nuevos = ['Abascal', 'Abel Caballero', 'Ada Colau', 'Adam Przeworski', 'Adolfo Suárez', 'Adriana Lastra', 'Adrián Vázquez', 'Agustín Moreno', 'Aitor Martínez', 'Albert Boadella', 'Albert Comellas', 'Albert Garcia', 'Albert Noguera', 'Albert Rivera', 'Alberto Alcocer', 'Alberto Casero', 'Alberto Fernández', 'Alberto Garzón', 'Alberto Luceño', 'Alberto Núñez Feijóo', 'Alberto Reyero', 'Alberto Rodríguez', 'Alberto San Juan', 'Alejandra Jacinto', 'Alfonso Bauluz', 'Alfonso Grau', 'Alfonso Guerra', 'Alfonso Rojo', 'Alfonso Rueda', 'Alfonso Serrano', 'Alfredo Montoya', 'Alicia Sánchez-Camacho', 'Ana Pontón', 'Ana Rosa Quintana', 'Ana Torroja', 'Ana Varela', 'Ander Gil', 'Anna Erra', 'Anna Gabriel', 'Anne Hidalgo', 'Antonia Morillas', 'Antonio Cerdá', 'Antonio Fernández', 'Antonio Garamendi', 'Antonio Hernando', 'Antonio Miguel Carmona', 'Antonio Muñoz', 'Antonio Vergara', 'Antonio de la Torre', 'Aragonès', 'Arias Navarro', 'Arkaitz Rodríguez', 'Arnaldo Otegi', 'Arturo González Panero', 'Aurora Picornell', 'Ayuso', 'Aznar', 'Bakartxo Ruiz', 'Baltasar Garzón', 'Barbón', 'Beatriz Gimeno', 'Begoña Gómez', 'Belarra', 'Bernie Sanders', 'Blas Infante', 'Blas de Lezo', 'Bolaños', 'Boluarte', 'Boris Johnson', 'Borja San Ramón', 'Borja Sémper', 'Brahim Ghali', 'Bárcenas', 'Carles Puigdemont', 'Carlos Arenas', 'Carlos Bardem', 'Carlos Fabra', 'Carlos García Adanero', 'Carlos García Juliá', 'Carlos Lesmes', 'Carlos Martín', 'Carlos Martínez-Almeida', 'Carlos Mazón', 'Carlos Prieto', 'Carlos Rojas', 'Carlos Sánchez Mato', 'Carlota Sales', 'Carmen Calvo', 'Carmen Fúnez', 'Carmen Martínez Aguayo', 'Carolina Alonso', 'Carolina Darias', 'Carolina Elías', 'Carrero Blanco', 'Casado', 'Celia Cánovas', 'Celia Mayer', 'Celia Villalobos', 'Cipriano Martos', 'Clara Campoamor', 'Clara Ponsatí', 'Colón de Carvajal', 'Corinna Larsen', 'Cosidó', 'Cospedal', 'Cristina Cifuentes', 'Cristina Fallarás', 'Cristina Fernández', 'Cristina Fernández de Kirchner', 'Cristina Narbona', 'Cristina Seguí', 'Cristina Zalba', 'Cuca Gamarra', 'Cuixart', 'Cándido Conde-Pumpido', 'Daniel Guzmán', 'Daniel Ripa', 'Daniel Sirera', 'David Beriain', 'David Leo', 'David Lucas', 'David Moya', 'David Simon', 'David Suárez', 'David Urdín', 'David de la Cruz', 'Delcy Rodríguez', 'Diana Morant', 'Diana Riba', 'Diego Cañamero', 'Dina Bousselham', 'Dióscoro Galindo', 'Dolores Delgado', 'Dolors Sabater', 'Donald Trump', 'Díaz Ayuso', 'Edmundo Bal', 'Eduard Pujol', 'Eduardo Rubiño', 'Eduardo Zaplana', 'Elías Bendodo', 'Emilio Hellín', 'Enrique Arnaldo', 'Enrique Cayuela', 'Enrique López', 'Enrique Ossorio', 'Enrique Santiago', 'Enriqueta Chicano', 'Ernest Maragall', 'Ernest Urtasun', 'Ernesto Alba', 'Esperanza Aguirre', 'Esperanza Casteleiro', 'Esperanza Gómez', 'Esteban', 'Esteban Beltrán', 'Esteban González Pons', 'Esteban Álvarez', 'Esther López', 'Eugenio Merino', 'Eugenio Pino', 'Eva Granados', 'Eva Kaili', 'Fadel Breica', 'Fatima Hamed', 'Fausto Canales', 'Feijoo', 'Feijoó', 'Felipe Alcaraz', 'Felipe González', 'Felipe Sicilia', 'Felipe VI', 'Fermín Muguruza', 'Fernandez Díaz', 'Fernando Barcia', 'Fernando Plaza', 'Fernando Presencia', 'Fernando Valdés', 'Fernández Díaz', 'Fernández Vara', 'Florentino Pérez', 'Forcadell', 'Fran Ferri', 'Francesc Vallès', 'Francisco', 'Francisco Camps', 'Francisco Correa', 'Francisco Cuenca', 'Francisco Franco', 'Francisco Fuentes', 'Francisco González', 'Francisco Igea', 'Francisco Martín', 'Francisco Martínez', 'Franco', 'Fèlix Larrosa', 'Fèlix Millet', 'Félix Bolaños', 'Félix López Rey', 'Gabriel Boric', 'Gabriel Rufián', 'Gallardo', 'Galán', 'Gandía Arturo Torró', 'Gara Santana', 'Georg Michael Welzel', 'Gerardo Iglesias', 'Gerardo Pisarello', 'Gijón', 'Gisbert', 'Gloria Calero', 'Gloria Elizo', 'Gloria Martín', 'Gloria Steinem', 'Gonzalo Berger', 'Gonzalo Boye', 'Gonzalo Caballero', 'Gonzalo Capellán', 'Gonzalo Pérez Jácome', 'Gonzalo Santonja', 'González Laya', 'González Pons', 'Gregorio Ordóñez', 'Griñán', 'Guillermo Fernández Vara', 'Gustavo Petro', 'Hana Jalloul', 'Helena Maleno', 'Héctor Gómez', 'Héctor Illueca', 'Ian Gibson', 'Iglesias', 'Ignacio Aguado', 'Ignacio Cembrero', 'Ignacio González', 'Ignacio Ramonet', 'Ildefonso Falcones', 'Inma Nieto', 'Inmaculada Nieto', 'Inés Arrimadas', 'Inés Herreros', 'Ione Belarra', 'Irene Lozano', 'Irene Montero', 'Irune Costumero', 'Isa Serra', 'Isabel Bonig', 'Isabel Díaz Ayuso', 'Isabel II', 'Isabel Peralta', 'Isabel Rodríguez', 'Isidoro Moreno', 'Iván Duque', 'Jaume Asens', 'Jaume Collboni', 'Jaume Graells', 'Jaume Matas', 'Javier Arenas', 'Javier Ayala', 'Javier Biosca', 'Javier Imbroda', 'Javier Lambán', 'Javier Losada', 'Javier Madrazo', 'Javier Maroto', 'Javier Negre', 'Javier Ruiz', 'Jesús Aguirre', 'Jesús Domínguez', 'Jesús Gil', 'Jesús Jurado', 'Jiménez Losantos', 'Joan Baldoví', 'Joan Fuster', 'Joan Ribó', 'Joan Subirats', 'Joaquim Bosch', 'Joaquim Forn', 'Joaquín Leguina', 'Joaquín Prat', 'Joe Biden', 'Jon Iñarritu', 'Jordi Cuixart', 'Jordi Martí', 'Jordi Puigneró', 'Jordi Pujol', 'Jordi Solé', 'Jordi Sànchez', 'Jordi Évole', 'Jorge Azcón', 'Jorge Fernández Díaz', 'Jorge Ignacio Palma', 'Josep Borrell', 'Josep Bou', 'Josep Maria Estela', 'Josep Maria Montaner', 'Josep Piqué', 'Josep Pujol', 'Josep Rius', 'Josep Rull', 'Josep Sunyol', 'José Antonio Griñán', 'José Antonio Nieto', 'José Barrionuevo', 'José Bernal', 'José Bono', 'José García Buitrón', 'José Ignacio García', 'José Luis Moreno', 'José Luis Peñas', 'José Luis Ábalos', 'José Manuel Bandrés', 'José Manuel Franco', 'José Manuel Miñones', 'José Manuel Soria', 'José María Cataluña', 'José María González', 'José Miñones', 'José Monzo', 'José Primo de Rivera', 'José de Francisco', 'Juan Antonio Delgado', 'Juan Bernardo Fuentes', 'Juan Bravo', 'Juan Carlos', 'Juan Carlos Aparicio', 'Juan Carlos Campo', 'Juan Carlos García Goena', 'Juan Carlos Girauta', 'Juan Carlos I', 'Juan Carlos Monedero', 'Juan Enciso', 'Juan Espadas', 'Juan García-Gallardo', 'Juan Ignacio Campos', 'Juan Lobato', 'Juan Marín', 'Juan Miguel Villar Mir', 'Juan Pedro Yllanes', 'Juan Rodríguez Poo', 'Juan de la Cierva', 'Juana Rivas', 'Juana Ruiz', 'Juanjo Carmona', 'Juanma Guijo', 'Juanma Moreno', 'Juanma del Olmo', 'Julian Assange', 'Julio Anguita', 'Julián Grimau', 'Julián Martínez', 'Junqueras', 'Justa Freire', 'Lambán', 'Largo Caballero', 'Laura Caldito', 'Laura Díez', 'Laura Luelmo', 'Laura Martín', 'Leggeri', 'Lenin', 'Leonor Paqué', 'Leonor de Borbón', 'Leopoldo López', 'Leticia Sabater', 'Letizia', 'Letizia Ortiz', 'Lidia Rubio', 'Lilith Verstrynge', 'Lizz Truss', 'Lluc Salellas', 'Lluis Apesteguia', 'Lluís Apesteguia', 'Lluís Rabell', 'Lluïsa Moret', 'Lorena Roldán', 'Lourdes Cebollero', 'Lourdes Maldonado', 'Lucía Figar', 'Luis Bárcenas', 'Luis Cueto', 'Luis García Montero', 'Luis Medina', 'Luis Olivera', 'Luis Tudanca', 'Luis Ángel Hierro', 'López Aguilar', 'López Obrador', 'Macarena Olona', 'Madrid Beltrán Gutiérrez', 'Manolo Mata', 'Manu Castro', 'Manu Pérez', 'Manuel Castells', 'Manuel Clavero Arévalo', 'Manuel Fraga', 'Manuel Gerena', 'Manuel Lapeña', 'Manuel Marchena', 'Manuel Prado', 'Manuel Ruiz', 'Manuela Bergerot', 'Manuela Carmena', 'Mar García Puig', 'Marcel Moore', 'Marcial Dorado', 'Marcos Ana', 'Marga Ferré', 'Margarita Robles', 'Mariano Rajoy', 'Mariano Sánchez Soler', 'Mariano Veganzones', 'Maribel Mora', 'Mario Vaquerizo', 'Marta Calvo', 'Marta Higueras', 'Marta Rovira', 'Marta del Castillo', 'Martina Velarde', 'Juanma Moreno', 'Martín González', 'Martín Villa', 'Maru Díaz', 'María Asencio', 'María Barrero', 'María Botto', 'María González Veracruz', 'María Guardiola', 'María Marín', 'María Rozas', 'María Teresa Fernández de la Vega', 'María Teresa Pérez', 'Matilde Eiroa', 'Mauricio Casals', 'Maza', 'Melisa Rodríguez', 'Mercedes González', 'Mercedez González', 'Mercé Gironés', 'Meri Pita', 'Miguel Anxo Fernández Lores', 'Miguel Buch', 'Miguel Campos', 'Miguel Delibes', 'Miguel González', 'Miguel Hernández', 'Miguel Urbán', 'Miguel Ángel Blanco', 'Miguel Ángel Rodríguez', 'Mikel Antza', 'Mikel Torres', 'Mikel Zabalza', 'Millán Astray', 'Miquel Pueyo', 'Mireia Comas', 'Mireia Vehí', 'Mohamed VI', 'Monago', 'Monedero', 'Montero', 'Montesinos', 'Montás', 'Mónica García', 'Mónica González', 'Móstoles', 'Nacho Calle', 'Nadia Calviño', 'Naim Darrechi', 'Nancy Pelosi', 'Narcís Serra', 'Narváez', 'Ni Belarra', 'Nieto Castro', 'Néstor Rego', 'Núria Marín', 'Obama', 'Odón Elorza', 'Olatz Vázquez', 'Oriol Junqueras', 'Ortega Cano', 'Ortega Smith', 'Ossorio', 'Pablo Bustinduy', 'Pablo Casado', 'Pablo Crespo', 'Pablo Echenique', 'Pablo Fernández', 'Pablo González', 'Pablo Hasel', 'Pablo Ibar', 'Pablo Iglesias', 'Pablo Montesinos', 'Pablo Motos', 'Pablo Zapatero', 'Paco Bezerra', 'Paco Espinosa', 'Paco Vázquez', 'Pacto de Toledo', 'Paloma Alonso', 'Paloma Fernández Coleto', 'Pamela Palenciano', 'Paqui Maqueda', 'Pardo Bazán', 'Pasqual Maragall', 'Patricia Guasp', 'Patricio P. Escobal', 'Pau Fons', 'Pau Ricomà', 'Paz Esteban', 'Pedraz', 'Pedro Antonio Sánchez', 'Pedro Arriola', 'Pedro Campos', 'Pedro Castillo', 'Pedro Cortés', 'Pedro González-Trevijano', 'Pedro Mouriño', 'Pedro Quevedo', 'Pedro Sanz', 'Pedro Sánchez', 'Pedro Varela', 'Pedro del Cura', 'Pepe Vélez', 'Pepe Álvarez', 'Pepu Hernández', 'Pilar Alegría', 'Pilar Garrido', 'Pilar Lima', 'Pilar Llop', 'Plácido Domingo', 'Pío García Escudero', 'Quim Torra', 'Rafael Vera', 'Rajoy', 'Ramiro de Maeztu', 'Ramón Luis Valcárcel', 'Ramón Tamames', 'Ramón de Carranza', 'Raquel Sánchez', 'Raúl Solís', 'Raül Blanco', 'Raül Romeva', 'Repsol Suárez', 'Revilla', 'Rey Juan Carlos', 'Reyes Maroto', 'Ribó', 'Ricardo Melchior', 'Rita Barberá', 'Rita Maestre', 'Rivera', 'Roberto Fraile', 'Roberto Lakidain', 'Roberto Saviano', 'Roberto Sotomayor', 'Robles', 'Rocío Delgado', 'Rocío Monasterio', 'Rodrigo Cuevas', 'Rodrigo Rato', 'Rodrigo Torrijos', 'Rodríguez Fraga', 'Rodríguez Palop', 'Roger Torrent', 'Roldán', 'Rosa Díez', 'Rosa García Alcón', 'Rosa Pérez', 'Rosa Pérez Garijo', 'Rosalía Iglesias', 'Royuela', 'Ruiz de Gordoa', 'Rupérez', 'Ruth Porta', 'Rutte', 'Salvador Alba', 'Salvador Illa', 'Salvador Martín Valdivia', 'Samuel Luiz', 'Sánchez', 'San Chin Choon', 'San Juan Nepomuceno', 'Sandra Gómez', 'Sandra Heredia', 'Santiago Abascal', 'Santos Cerdán', 'Scott Morrison', 'Sebastian Kurz', 'Sergio Gómez Reyes', 'Sofía Castañón', 'Sol Sánchez', 'Soledad Castillero', 'Soledad Luque', 'Soledad Murillo', 'Sonia Vivas', 'Susana Díaz', 'Susana Hornillo', 'Suso Díaz', 'Tania Sánchez', 'Tania Varela', 'Teo García Egea', 'Teodoro García Egea', 'Teresa Bueyes', 'Teresa Jiménez Becerril', 'Teresa Ribera', 'Teresa Rodríguez', 'Teresa Rodríguez Rubio', 'Tito Berni', 'Tomás Díaz Ayuso', 'Toni Cantó', 'Toni Morillas', 'Toni Nadal', 'Toni Rodon', 'Toni Valero', 'Tornero', 'Torres', 'Trias', 'Unai Sordo', 'Unai Urruzuno', 'Utrera Molina', 'Vallejo', 'Vargas Llosa', 'Vicente Lertxundi', 'Vicente del Bosque', 'Vicenç Villatoro', 'Vicky Rosell', 'Victoria Landa', 'Victoria Rosell', 'Videla', 'Víctor Gutiérrez', 'Víznar', 'Waldino Varela', 'Willy Toledo', 'Xavier Domènech', 'Xavier García Albiol', 'Xavier Rius', 'Xavier Trias', 'Ximo Puig', 'Xiomara Castro', 'Xosé Sánchez Bugallo', 'Yassin Kanjaa', 'Yeremi Vargas', 'Yolanda Diaz', 'Zapatero', 'Zelenski', 'Francisco Vázquez', 'Paco Vázquez', 'Abel Caballero', 'Ada Colau', 'Adolfo Suárez', 'Adolfo Suárez Illana', 'Adriana Lastra', 'Adrián Barbón', 'Aitor Esteban', 'Albert Boadella', 'Albert Rivera', 'Alberto Garzón', 'Alberto Ruiz-Gallardón', 'Alfonso Alonso', 'Alfonso Guerra', 'Alfredo Pérez Rubalcaba', 'Alicia Sánchez-Camacho', 'Ana Botella', 'Ana Botín', 'Ana Mato', 'Ana Oramas', 'Ana Palacio', 'Ana Pastor', 'Ana Rosa Quintana', 'Ander Gil', 'Andrea Levy', 'Anna Gabriel', 'Antoni Comín', 'Antonio Baños', 'Antonio Hernando', 'Antonio Machado', 'Antonio Maíllo', 'Antonio Sanz', 'Arancha González Laya', 'Artur Mas', 'Baltasar Garzón', 'Beatriz Corredor', 'Begoña Gómez', 'Belarra', 'Bermúdez de Castro', 'Bono', 'Borja Sémper', 'Borrell', 'Boti García', 'Boya', 'Bousselham', 'Bétera', 'Calviño', 'Calvo', 'Camilo de Dios', 'Cani Fernández', 'Carla Antonelli', 'Carles Campuzano', 'Carles Mulet', 'Carles Puigdemont', 'Carme Forcadell', 'Carmela Silva', 'Carmen Alborch', 'Carmen Calvo', 'Carmen Castilla', 'Carmen Franco Polo', 'Carmen Martínez-Bordiú', 'Carmen Montón', 'Carmen Polo', 'Carmen Torres', 'Carmen de la Peña', 'Carolina Darias', 'Casado', 'Abascal', 'Cayetana Álvarez de Toledo', 'Celia Villalobos', 'Clara Campoamor', 'Cristina Cifuentes', 'Cs Ángel Garrido', 'Díaz Ayuso', 'Díez Picazo', 'Edmundo Bal', 'Fernando Clavijo', 'Fernando Martín', 'Fernando Martínez López', 'Fernando Miramontes', 'Fernando Roig', 'Fernando Román', 'Fernando Savater', 'Fernando Simón', 'Fernando Valdés', 'Fernández Díaz', 'Fernández Vara', 'Florentino Pérez', 'Francis Franco', 'Francisco Camps', 'Francisco Correa', 'Francisco Franco', 'Francisco González', 'Francisco Granados', 'Francisco Igea', 'Francisco Javier Guerrero', 'Francisco Javier Sánchez Gil', 'Francisco Martínez', 'Francisco Serrano', 'Francisco Vázquez', 'Gabriel Rufián', 'Gaspar Llamazares', 'Gerardo Iglesias', 'Gonzalo Caballero', 'González Laya', 'González Pons', 'Griñán', 'Guindos', 'Gómez', 'Gómez-Reino', 'Hana Jalloul', 'Ignacio Aguado', 'Ignacio Escolar', 'Ignacio Garriga', 'Ignacio González', 'Iglesias', 'Inés Arrimadas', 'Ione Belarra', 'Iratxe García', 'Irene Lozano', 'Irene Montero', 'Isa Serra', 'Isabel Díaz Ayuso', 'Iturgaiz', 'Iván Espinosa de los Monteros', 'Iñigo Errejón', 'Iñigo de la Serna', 'Javier Arenas', 'Javier Maroto', 'Javier Nart', 'Javier Negre', 'Javier Ortega Smith', 'Javier Solana', 'Javier Zarzalejos', 'Jenaro Castro', 'Jenn Díaz', 'Jenner López Escudero', 'Jesús Montero', 'Jesús Muñecas', 'Jesús Sepúlveda', 'Joan Baldoví', 'Joan Coscubiela', 'Joan Garcés', 'Joan Herrera', 'Joan Mena', 'Joan Mesquida', 'Joan Ribó', 'Joan Subirats', 'Joaquim Forn', 'Joaquín Leguina', 'Joaquín Pérez Rey', 'Jordi Alemany', 'Jordi Borràs', 'Jordi Cuixart', 'Jordi Montull', 'Jordi Pujol', 'Jordi Pujol Ferrusola', 'Jordi Salvador', 'Jordi Sevilla', 'Jordi Sànchez', 'Jordi Turull', 'Jordi Xuclà', 'Jorge Azcón', 'Jorge Fernández Díaz', 'Joseba Pagazaurtundúa', 'Josep Borrell', 'Josep Lluis Núñez', 'Josep Lluis Trapero', 'Josep Piqué', 'Josep Rull', 'Josu Erkoreka', 'José Antonio Sánchez', 'José Blanco', 'José Bono', 'José Couso', 'José Guirao', 'José Luis Ábalos', 'José María Aznar', 'José María García', 'José María Marco', 'José Ramón Bauzá', 'José Ángel Fernández Villa', 'Juan Carlos Campo', 'Juan Carlos Girauta', 'Juan Carlos I', 'Juan Carlos Monedero', 'Juan Carlos de Borbón', 'Juan Cotino', 'Juan Espadas', 'Juan Genovés', 'Juan Guaidó', 'Juan José Cortes', 'Juan José Tamayo', 'Juan Luis Cebrián', 'Juan Luis Rubenach', 'Juan María González', 'Juan Marín', 'Juan Muñoz', 'Juan Romero', 'Juan Rosell', 'Juan Trinidad', 'Juanma Moreno', 'Juanma Romero', 'Juanma Serrano', 'Julio Anguita', 'Julio Rodríguez', 'Junqueras', 'Jéssica Albiach', 'Lamela', 'Largo Caballero', 'Largo Mayo', 'Laura Borrás', 'Laura Duarte', 'Leopoldo López', 'Luis Bárcenas', 'Luis De Guindos', 'Manuel Azaña', 'Manuel Fraga', 'Manuela Carmena', 'Mariano Rajoy', 'Marta Pascal', 'Marta Rivera de la Cruz', 'Marta Rovira', 'Martín Villa', 'Martínez Almeida', 'María Teresa Fernández de la Vega', 'Melisa Rodríguez', 'Meritxel Batet', 'Miguel Ángel Blanco', 'Miguel Ángel Revilla', 'Miguel Ángel Rodríguez', 'Milagrosa Martínez', 'Miquel Buch', 'Miquel Iceta', 'Miquel Roca', 'Mónica García', 'Nadia Calviño', 'Narcís Serra', 'Núria Marín', 'Oriol Junqueras', 'Oriol Pujol', 'Ortega Smith', 'Otegi', 'Pablo Casado', 'Pablo Iglesias', 'Pablo Echenique', 'Pablo Hasel', 'Pablo Ibar', 'Paco Ferrándiz', 'Paco Frutos', 'Paco Guarido', 'Pedro Duque', 'Pedro García Aguado', 'Pedro J. Ramírez', 'Pedro Quevedo', 'Pedro Rollán', 'Pedro Santisteve', 'Pedro Sánchez', 'Pepe Mujica', 'Pepu Hernández', 'Pere Navarro', 'Pilar Rahola', 'Piqué', 'Primo de Rivera', 'Pujol', 'Pío García Escudero', 'Quim Forn', 'Quim Torra', 'Rajoy', 'Ramón Espinar', 'Ramón Jáuregui', 'Raquel Martínez', 'Raquel Romero', 'Rita Barberá', 'Rita Maestre', 'Roberto Fraile', 'Rocío Monasterio', 'Rodrigo Rato', 'Rosa Díez', 'Ruth Beitia', 'Salvador Illa', 'Santiago Abascal', 'Santiago Vidal', 'Susana Díaz', 'Soraya Rodríguez', 'Soraya Sáenz', 'Teresa Ribera', 'Teodoro García Egea', 'Uxue Barkos', 'Valerio', 'Víctor Barrio', 'Xabier Arzalluz', 'Xavi Hernández', 'Xavier Domènech', 'Xavier García Albiol', 'Ximo Puig', 'Xosé Manuel Beiras', 'Xulio Ferreiro', 'Yolanda Díaz', 'Yolanda González', 'Zapatero', 'Zerolo', 'Àngel Ros', 'Álvaro Lapuerta', 'Ángel Acebes', 'Ángel Gabilondo', 'Ángel Garrido', 'Ángel Viñas', 'Ángel Víctor Torres', 'Ángela Aguilera', 'Ángeles Pedraza', 'Íñigo Errejón']

partidos_originales = ['Partido Socialista Obrero Español', 'PSOE', 'Podemos', 'Partido Popular', 'PP', 'ERC', 'Esquerra Republicana de Catalunya', 'Ciudadanos', 'UPN', 'Unión del Pueblo Navarro', 'PSC', 'Partit dels Socialistes de Catalunya', 'IU', 'Izquierda Unida', 'EAJ-PNV', 'Compromís', 'Barcelona En Común', 'EQUO', 'Iniciativa vers per Catalunya', 'Coalición Canaria', 'EH Bildu', 'Bildu', 'Euskal Herria Bildu', 'Esquerra Unida i Alternativa', 'BNG', 'Bloque Nacionalista Galego', 'PDeCat', 'PACMA', 'Vox', 'VOX', 'NCa', 'Nueva Canarias', 'MÉS per Mallorca', 'FAC', 'Foro Asturias', 'PNC', 'Partido Nacionalista Canario', 'ASG', 'Ciudadanos']

partidos_nuevos = ['PSOE', 'Podemos', 'PP', 'Esquerra Republicana de Catalunya', 'Ciudadanos', 'UPN', 'PSC', 'Izquierda Unida', 'EAJ-PNV', 'Compromís', 'Barcelona En Común', 'EQUO', 'Iniciativa vers per Catalunya', 'Coalición Canaria', 'Bildu', 'Esquerra Unida i Alternativa', 'BNG', 'PDeCat', 'PACMA', 'Vox', 'VOX', 'Nueva Canarias', 'MÉS per Mallorca', 'FAC', 'Foro Asturias', 'PNC', 'ASG', 'Ciudadanos']

modificar_dataset(dataset_path, nombres_originales, nombres_nuevos, partidos_originales, partidos_nuevos, resultado_modificado, resultado_no_modificado)