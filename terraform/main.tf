# Configure the AWS provider
provider "aws" {
  access_key = var.access_key
  secret_key = var.secret_key
  region     = var.region
}


# Create an EC2 instance
resource "aws_instance" "web_server" {
  ami           = "ami-0b33d91d"
  instance_type = "t2.micro"

  # Add storage to the instance
  root_block_device {
    volume_size = "8"
  }

  # Add a security group to the instance
  vpc_security_group_ids = [aws_security_group.sg.id]

  # Add a public IP address to the instance
  associate_public_ip_address = true
}

# Create a security group
resource "aws_security_group" "sg" {
  name        = "web_server_sg"
  description = "Security group for web server"

  # Allow incoming HTTP traffic
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_s3_bucket" "example" {
  bucket = "my-tf-example-bucket"
}

resource "aws_s3_bucket_acl" "example_bucket_acl" {
  bucket = aws_s3_bucket.example.id
  acl    = "private"
}

resource "aws_s3_bucket_object" "object" {
  bucket = aws_s3_bucket.example.id
  key    = "model.h5"
  source = "C:/Users/victo/Desktop/Cours/Projet-Cloud-Detect/Clouds_Detector/my_checkpoint.pth.h5"

  # The filemd5() function is available in Terraform 0.11.12 and later
  # For Terraform 0.11.11 and earlier, use the md5() function and the file() function:
  # etag = "${md5(file("path/to/file"))}"
  etag = filemd5("C:/Users/victo/Desktop/Cours/Projet-Cloud-Detect/Clouds_Detector/my_checkpoint.pth.h5")
}
