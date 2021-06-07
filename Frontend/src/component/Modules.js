import React from "react";
import {List, Avatar, Icon, Button} from "antd";
import {Link} from "react-router-dom";
import axios from "axios";
const IconText = ({ type, text }) => (
  <span>
    <Icon type={type} style={{ marginRight: 8 }} />
    {text}
  </span>
);
class Modules extends React.Component {
    constructor(props)
    {
      super(props)
      this.state = {
        modules: [],
        users: [],
        profile: []
      }
    }
    componentDidMount(){
      axios.all([
        axios.get('http://127.0.0.1:8000/api/users'),
        axios.get('http://127.0.0.1:8000/api/list/profile'),
    ])
    .then(axios.spread((users, profile) => {
            this.setState({
                users: users.data,
                profile: profile.data,
            })
            console.log(this.state.users)
            console.log(this.state.profile)
            
    }))
    }
    examCreate = () => {
        this.setState({
          confirmed: true,
        })
    }
    testFunction = (e) => {
      
      console.log(e.title);
    }
    render(){
        let userID = this.props.user
        console.log(userID)
        let user_data = []
        let user_id = 0
        let temp_modules = []
        let module_data = this.props.data
        
        axios.get('http://127.0.0.1:8000/api/users')
        .then(res => {
            user_data = res.data
            user_data.map(function(item, i){
                if(item.username == userID)
                {
                    console.log("yes")
                    user_id = item.id
                    
                  
                    module_data.map(function(mod, i){
                      if(mod.user == user_id){
                          temp_modules.push(mod)         
                      }
                    })
                }
            })
           console.log(temp_modules[0])
           
           this.setState({
             modules: temp_modules
           })
        })
        let tempUsers = this.state.users
        let tempProfile = this.state.profile

        
        
        let id = 0
        let profile_picture = ""
        console.log(id)
        tempUsers.map(function(item, i){
            if(item.username == userID)
            {
              id = item.id
            } 
            
        })
        
        tempProfile.map(function(item, i){
            if(item.user == id)
            {
                profile_picture = item.avatar;
            }
              
        })
      return (
          <List
          itemLayout="vertical"
          size="large"
          pagination={{
            onChange: page => {
              console.log(page);
            },
            pageSize: 3, 
          }}
          dataSource={this.state.modules}
          renderItem={item => (
            <List.Item
              key={item.title}
              actions={[
                <IconText type="star-o" text="156" key="list-vertical-star-o" />,
                <IconText type="like-o" text="156" key="list-vertical-like-o" />,
                <IconText type="message" text="2" key="list-vertical-message" />,
              ]}
              extra={
                <div>
                  <Link to={`/createExam/` + item.id + '/' + this.props.user}>
                    <Button onClick={this.examCreate}>
                        Create a Test
                    </Button> 
                  </Link>
                </div>
              }
            >
              Code: {item.code}
              <List.Item.Meta
                avatar={<Avatar src={profile_picture} />}
                title={<Link to={`/modules/${item.id}/${this.props.user}`}>{item.title}</Link>}
                description={item.description}
                num={item.num_students}
              />
              {item.content}
            </List.Item>
          )}
        />
      )
  }
}
export default Modules
