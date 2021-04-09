<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Vet Clinic-Your Pets</title>
</head>
<body>
<?php
   include 'connectdb.php';
?>
<ol>
<?php
   $whichOwner= $_POST["petowners"];
   $petName = $_POST["petname"];
   $species =$_POST["species"];
   $query1= 'select max(petid) as maxid from pet';
   $result= $connection->query($query1);
   $row=$result->fetch();
   $newkey = intval($row["maxid"]) + 1;
   $petid = (string)$newkey;
   $query2 = 'INSERT INTO pet values("' . $petid . '","' . $petName . '","' . $species . '","' . $whichOwner . '")';
   $numRows = $connection->exec($query2);
   echo "Your pet was added!";
   $connection = NULL;
?>
</ol>
</body>
</html>