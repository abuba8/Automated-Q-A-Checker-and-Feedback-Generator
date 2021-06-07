import React from "react";
import {List, Avatar, Icon, Button} from "antd";
const IconText = ({ type, text }) => (
  <span>
    <Icon type={type} style={{ marginRight: 8 }} />
    {text}
  </span>
);
class Tests extends React.Component {
    constructor(props)
    {
      super(props)
      this.state = {
        confirmed: false,
      }
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
          dataSource={this.props.data}
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
                  <Button onClick={this.examCreate}>
                      Create An Exam
                  </Button> <br></br><br></br>
                  <Button onClick={() => this.testFunction(item)}>
                    See Test
                  </Button>
                </div>
              }
            >
              Code: {item.code}
              <List.Item.Meta
                avatar={<Avatar src={item.avatar} />}
                title={<a href={`/${item.id}`}>{item.title}</a>}
                description={item.description}
              />
              {item.content}
            </List.Item>
          )}
        />
      )
  }
}
export default Tests
