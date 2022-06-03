//Final Project12
//Program To Manage a Hotel


#include<iostream.h>
#include<conio.h>
#include<iomanip.h>
#include<dos.h>
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<fstream.h>


class hotel
     {
      public:
      int mainmenu();
      int bookroom();
      int showrecord();
      int deleterecord();
      int modifyrecord();
     };
hotel hotel1;

class customers
       {
	public:
	char name[30];
	long int phone;
	char email[400];
	int room;

	int enter();
	void display();
	int emailchecker(char arr[]);

		       };
customers person;


int width=40;

int main()
{


hotel1.mainmenu();

getch();
return 0;
}

int hotel::mainmenu()
    {
     int choice;


     for(int depth=1;depth<=5;depth++)
	cout<<endl;

     cout<<setw(width+3)<<"-----------"<<endl;
     cout<<setw(width+2)<<"MAIN MENU"<<endl;
     cout<<setw(width+3)<<"-----------"<<endl;

     cout<<endl<<endl;

     cout<<setw(width+3)<<"1)Book A Room"<<endl;
     cout<<setw(width+8)<<"2)Customer Records"<<endl;
     cout<<setw(width+6)<<"3)Delete Records"<<endl;
     cout<<setw(width+4)<<"4)Edit Records"<<endl;
     cout<<setw(width-4)<<"5)Exit"<<endl;

     cout<<endl;

     cout<<setw(width+7)<<"Enter Your Choice : ";
     cin>>choice;

     switch(choice)
	   {
	    case 1:
	    cout<<setw(width+7)<<"Book A Room Selected";
	    delay(600);
	    system("CLS");
	    bookroom();
	    break;

	    case 2:
	    cout<<setw(width+11)<<"Customer Records Selected";
	    delay(600);
	    system("CLS");
	    showrecord();
	    break;

	    case 3:
	    cout<<setw(width+9)<<"Delete Record Selected";
	    delay(600);
	    system("CLS");
	    deleterecord();
	    break;

	    case 4:
	    cout<<setw(width+10)<<"Modify Records Selected";
	    delay(600);
	    system("CLS");
	    modifyrecord();
	    break;

	    case 5:
	    cout<<setw(width+8)<<"Press Any Key To Exit";
	    break;

	    default:
	    cout<<setw(width+12)<<"Please enter valid Number";
	    delay(600);
	    system("CLS");
	    mainmenu();


					    }


     return 0;

					     }

int hotel::bookroom()
   {
    int roomno;
    int flag=0;

    for(int depth=1;depth<=5;depth++)
	cout<<endl;

    cout<<setw(width+3)<<"-----------"<<endl;
    cout<<setw(width+2)<<"Book Room"<<endl;
    cout<<setw(width+3)<<"-----------"<<endl<<endl;

    person.enter();


    roomno=person.room;

    customers reader;
    ifstream file;

    file.open("CBSEH.dat",ios::in);
    file.seekg(0,ios::beg);


    while(!file.eof())
	 {
	  file.read( (char *)&reader,sizeof(reader) );

	  if(reader.room==roomno)
	    {
	     flag=1;
	     break;
	    }

	 }

    file.close();

    if(flag==1)
      {
       cout<<setw(width+18)<<"Room Has Already Been Booked";
       delay(600);
       system("CLS");
      }


    else if(flag==0)
      {
       ofstream file1;
       file1.open("CBSEH.dat",ios::app);
       file1.write( (char *) &person,sizeof(person) );
       file1.close();

       cout<<endl<<setw(width+15)<<"Record Stored Succesfully";
       delay(600);
      }

    system("CLS");
    mainmenu();
    return 0;
   }

int hotel::showrecord()
   {
     int choice1;
     int phoneno=-1;
     int roomnum=-1;
     int find=0;

     for(int depth=1;depth<=5;depth++)
	cout<<endl;

     cout<<setw(width+10)<<"------------------"<<endl;
     cout<<setw(width+9)<<"Customer Records"<<endl;
     cout<<setw(width+10)<<"------------------"<<endl;
     cout<<endl;
     cout<<setw(width+9)<<"1.Search By RoomNo"<<endl;
     cout<<setw(width+10)<<"2.Search By PhoneNo"<<endl;
     cout<<setw(width+11)<<"3.Return To MainMenu"<<endl;
     cout<<endl;
     cout<<setw(width+10)<<"Please Enter Choice : ";
     cin>>choice1;

     if(choice1==1)
       {
	cout<<setw(width+10)<<"Please Enter RoomNo : ";
	cin>>roomnum;
	phoneno=-1;
       }
     else if(choice1==2)
       {
	cout<<setw(width+11)<<"Please Enter PhoneNo : ";
	cin>>phoneno;
	roomnum=-1;
       }
     else if(choice1==3)
       {
	system("CLS");
	mainmenu();
	return 0;
       }

     else
       {
	cout<<setw(width+2)<<"Invalid Choice";
	delay(1000);
	system("CLS");
	showrecord();
	return 0;
       }

     system("CLS");

     for(depth=1;depth<=5;depth++)
	cout<<endl;

     cout<<setw(width+10)<<"------------------"<<endl;
     cout<<setw(width+9)<<"Customer Records"<<endl;
     cout<<setw(width+10)<<"------------------"<<endl;
     cout<<endl;


     fstream file;

     file.open("CBSEH.dat",ios::in);
     file.seekg(0,ios::beg);
     system("CLS");
     while(!file.eof())
	  {
	   file.read((char *) &person,sizeof(person));

	   if(person.room==roomnum&&phoneno==-1)
	     {
	      find=1;

	      break;
	     }
	   else if(person.phone==phoneno&&roomnum==-1)
	       {
		find=1;
		break;
	       }
	  }


     for(depth=1;depth<=5;depth++)
	cout<<endl;

     cout<<setw(width+10)<<"------------------"<<endl;
     cout<<setw(width+9)<<"Customer Records"<<endl;
     cout<<setw(width+10)<<"------------------"<<endl;
     cout<<endl;

     if(find==1)
       {
	person.display();
	delay(600);
	cout<<endl<<endl<<setw(width+20)<<"Please Press Any Key To Proceed...";
       }
     else
       {
	cout<<endl<<setw(width+7)<<"No Record Found";
	delay(600);
	cout<<endl<<endl<<setw(width+20)<<"Please Press Any Key To Proceed...";

       }

     file.close();
     getch();
     system("CLS");
     showrecord();
    return 0;
   }

int hotel::deleterecord()
   {
     int choice;
     int roomno,roomnoconfirm;

     for(int depth=1;depth<=5;depth++)
	cout<<endl;

     cout<<setw(width+7)<<"---------------"<<endl;
     cout<<setw(width+6)<<"Delete Record"<<endl;
     cout<<setw(width+7)<<"---------------"<<endl;


     cout<<endl<<endl<<setw(width+13)<<"1.Delete By Room Number "<<endl;
     cout<<setw(width+9)<<"2.Return To MainMenu"<<endl;
     cout<<endl<<endl<<setw(width+10)<<"Please Enter Choice : ";
     cin>>choice;

     if(choice==1)
       {
	system("CLS");
	for(int depth=1;depth<=5;depth++)
	cout<<endl;

	cout<<setw(width+7)<<"---------------"<<endl;
	cout<<setw(width+6)<<"Delete Record"<<endl;
	cout<<setw(width+7)<<"---------------"<<endl;

	cout<<endl<<setw(width+10)<<"Enter Room Number : ";
	cin>>roomno;
	cout<<setw(width+12)<<"Confirm Room Number : ";
	cin>>roomnoconfirm;
	if(roomno!=roomnoconfirm)
	  {
	   cout<<endl<<setw(width+16)<<"Room Number Does Not Match";
	   delay(600);
	   system("CLS");
	   deleterecord();
	   return 0;
	  }
	else
	  {

	fstream file1,file2;

	file1.open("CBSEH.dat",ios::in);
	file2.open("hotel2.dat",ios::out);

	while(!file1.eof())
	     {
	      file1.read((char *) &person,sizeof(person));

	      if(person.room!=roomno)
		{
		 file2.write((char *) &person,sizeof(person));
		}
	     }

	file1.close();
	file2.close();

	remove("CBSEH.dat");
	rename("hotel2.dat","CBSEH.dat");

       }    }

     else if(choice==2)
       {
	system("CLS");
	mainmenu();
	return 0;
       }
     else
       {
	cout<<setw(width+2)<<"Invalid Choice";
	delay(600);
	system("CLS");
	deleterecord();
	return 0;
       }

     cout<<endl<<endl<<setw(width+15)<<"Record Deleted Succesfully";
     delay(600);
     system("CLS");
    mainmenu();
    return 0;
   }

int hotel::modifyrecord()
   {

     for(int depth=1;depth<=5;depth++)
	cout<<endl;

     cout<<setw(width+7)<<"---------------"<<endl;
     cout<<setw(width+6)<<"Modify Record"<<endl;
     cout<<setw(width+7)<<"---------------"<<endl<<endl;

     int roomnum,roomconf,flag=0;
     long int pos;

     cout<<setw(width+15)<<"Please Enter Room Number : ";
     cin>>roomnum;



     fstream file;

     file.open("CBSEH.dat",ios::in|ios::out|ios::ate);
     file.seekg(0,ios::beg);

     while(!file.eof())
	  {
	   pos=file.tellg();
	   file.read((char *) &person,sizeof(person));

	   if(person.room==roomnum)
	     {
	      flag=1;
	      cout<<endl;
	      cout<<setw(width+12)<<"Please Enter New Details"<<endl;
	      cout<<setw(width+13)<<"--------------------------"<<endl;
	      person.enter();
	      cout<<setw(width+13)<<"--------------------------"<<endl;
	      cout<<endl<<setw(width+10)<<"Confirm Room Number : ";
	      cin>>roomconf;

	      if(roomconf==person.room)
		{
		 file.seekg(pos);
		 file.write((char *) &person,sizeof(person));
		 file.close();
		 cout<<endl<<setw(width+14)<<"Record Modified Successfuly";
		 delay(1200);
		 system("CLS");
		 mainmenu();
		 return 0;
		}
	      else
		{
		 cout<<setw(width+14)<<"Room Number Does Not Match";
		 file.close();
		 delay(1200);
		 system("CLS");
		 mainmenu();
		 return 0;
		}

	     }
	  }
	   if(flag==0)
	     {
	      cout<<setw(width+9)<<"Record Does Not Exist";
	      file.close();
	      delay(600);
	      system("CLS");
	      mainmenu();
	      return 0;
	     }


    mainmenu();
    return 0;
   }

int customers::enter()
	{
	 cout<<setw(width+10)<<"Enter Room Number : ";
	 cin>>person.room;

	 cout<<setw(width+3)<<"Enter Name : ";
	 gets(person.name);

	 cout<<setw(width+4)<<"Enter Email : ";
	 cin>>person.email;

	 if(!person.emailchecker(person.email))
	   {
	    cout<<setw(width+12)<<"WARNING! INVALID EMAIL"<<endl<<endl;
	   }

	 cout<<setw(width+11)<<"Enter Phone Number : ";
	 cin>>person.phone;

	 return 0;
	}

void customers::display()
    {
     cout<<setw(width)<<"Room No : "<<person.room<<endl;
     cout<<setw(width-3)<<"Name : "<<person.name<<endl;
     cout<<setw(width+5)<<"Phone Number : "<<person.phone<<endl;
     cout<<setw(width-2)<<"Email : "<<person.email;
    }

int customers::emailchecker(char arr[])
   {
    int flag=0;
    for(int mctr=0;arr[mctr]!=NULL;mctr++)
       {
	if(arr[mctr]=='@')
	  {
	   flag=1;
	   break;
	  }
       }
    return flag;
   }
