import React from "react";
import image from "../images/GradeMechanism.PNG"
class Grade extends React.Component {
    render()
    {
        return(
            <div>
                <img src={image} alt="Smiley face" width="100%" height="100%" />
            </div>
        )
    }
}
export default Grade