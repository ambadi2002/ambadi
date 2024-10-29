#!/usr/bin/env python
# coding: utf-8

# In[37]:


#connecting our database to python enviornment


# In[41]:


pip install pymysql


# In[42]:


import pymysql

connection = pymysql.connect(
    host='127.0.0.1',
    user='root',
    password='ambadibkurup@2024',
    database='library'
)
print(connection)


# In[45]:


#above conecting part can be done through a error handling method
#defining a function for it
#try except methode(error handling methode)


# In[46]:


def connect_to_database():
    #try is used to connect the database
    try:
      connection = pymysql.connect(
          host='127.0.0.1',
          user='root',
          password='ambadibkurup@2024',
          database='library'
      )
      return connection
#except is used to find error
    except pymysql.connect.Error as err:
        print("error connecting to database:",err)
        return none
    


# In[47]:


connection=connect_to_database()
print(connection)


# In[48]:


#adding a author to our data base


# In[57]:


def add_author(name):
    connection=connect_to_database()
    cursor=connection.cursor()
    #curser is an object that used to connect our python and sql
    sql="INSERT INTO author(name) VALUES (%s)"
    #values in %S is provided through data in next line
    data=(name,)#tuple with single variable
    cursor.execute(sql,data)
    connection.commit()
  


# In[58]:


add_author("bindu") 


# In[59]:


add_author("babu") 


# In[60]:


#inserting values to books table


# In[65]:


def add_book(title,authors_id,category,price):
    connection=connect_to_database()
    cursor=connection.cursor()
    #curser is an object that used to connect our python and sql
    sql="INSERT INTO books(title,authors_id,category,price) VALUES (%s,%s,%s,%s)"
    #values in %S is provided through data in next line
    data=(title,authors_id,category,price)#tuple with multiple variable
    cursor.execute(sql,data)
    connection.commit()


# In[66]:


add_book('ammamanas',7,'poem',15.99)


# In[67]:


add_book('daddy cool',8,'story',16.99)


# In[71]:


def get_all_books():
    connection=connect_to_database()
    cursor=connection.cursor()
    sql="SELECT * FROM books"
    cursor.execute(sql)
    results=cursor.fetchall()
    for row in results:
        print(f"title:{row[1]},authors_id:{row[2]},category:{row[3]},price:{row[4]}")


# In[73]:


get_all_books()


# In[74]:


#how to search


# In[79]:


def search_books(search_term):
    connection=connect_to_database()
    cursor=connection.cursor()
    sql="SELECT * FROM books INNER JOIN author ON author.id=books.authors_id WHERE title LIKE %s OR author.name LIKE %s"
    search_param=("%"+search_term+"%","%"+search_term+"%")
    cursor.execute(sql,search_param)
    results=cursor.fetchall()
    for row in results:
        print(f"title:{row[1]},authors_id:{row[2]},category:{row[3]},price:{row[4]}")


# In[82]:


search_books("daddy cool")


# In[86]:


search_books("ammamanas")


# In[87]:


search_books("babu")


# In[88]:


search_books("beema")


# In[ ]:




