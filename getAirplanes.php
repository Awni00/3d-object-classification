
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Schedule a Flight</title>
<link rel="stylesheet" href="mystyle.css">
</head>
<body>
<?php
include 'connectdb.php';
?>
<h1>Here are the available flights:</h1>
<table>
<?php
   $whichFlight = $_POST["flight"];
   $query = 'SELECT * FROM Flight WHERE Flight.FlightNumber="' . $whichFlight . '"';
   $result=$connection->query($query);
    while ($row=$result->fetch()) {
	echo "<tr><td>".$row["FlightNumber"]."</td><td>".$row["AirlineCode"]."</td></tr>";
    }
?>
</table>
<?php
   $connection = NULL;
?>
</body>
</html>